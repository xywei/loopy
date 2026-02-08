"""WGSL/WebGPU code generation target.

This is currently a *code-generation-only* target. It can generate WGSL device
code intended to be compiled by WebGPU implementations in browsers.

Execution support (i.e. an executor that dispatches the generated shaders via a
runtime such as wgpu/JS WebGPU) is intentionally not implemented yet.
"""

from __future__ import annotations


from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from typing_extensions import override

import pymbolic.primitives as p
from pymbolic import Expression, var
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.typing import ArithmeticExpression
from pytools import memoize_method

from loopy.diagnostic import LoopyError, LoopyTypeError
from loopy.expression import TypeContext, dtype_to_type_context
from loopy.kernel.data import AddressSpace, ArrayArg, ConstantArg, TemporaryVariable, ValueArg
from loopy.kernel.function_interface import ScalarCallable
from loopy.symbolic import GroupHardwareAxisIndex, LocalHardwareAxisIndex
from loopy.target import ASTBuilderBase, DummyHostASTBuilder, TargetBase
from loopy.target.c import DTypeRegistry
from loopy.target.c.codegen.expression import CExpressionToCodeMapper, ExpressionToCExpressionMapper
from loopy.types import LoopyType, NumpyType


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from loopy.codegen import CodeGenerationState, PreambleInfo
    from loopy.codegen.result import CodeGenerationResult
    from loopy.kernel import LoopKernel
    from loopy.kernel.instruction import Assignment
    from loopy.translation_unit import CallablesInferenceContext, CallablesTable
    from loopy.translation_unit import TranslationUnit


# {{{ dtype registry


@dataclass(frozen=True)
class WGSLDTypeRegistry(DTypeRegistry):
    """Minimal dtype registry mapping :class:`~loopy.types.LoopyType` to WGSL."""

    @override
    def get_or_register_dtype(self,
                              names: Sequence[str],
                              dtype: LoopyType | None = None) -> np.dtype[np.generic]:
        if dtype is None:
            raise LoopyTypeError("WGSLDTypeRegistry requires an explicit dtype")

        if isinstance(dtype, NumpyType):
            return dtype.numpy_dtype

        raise LoopyTypeError(f"unsupported loopy dtype for WGSL: {dtype!r}")

    @override
    def dtype_to_ctype(self, dtype: LoopyType) -> str:
        if not isinstance(dtype, NumpyType):
            raise LoopyTypeError(f"unsupported loopy dtype for WGSL: {dtype!r}")

        nd = dtype.numpy_dtype

        if nd == np.dtype(np.float32):
            return "f32"
        if nd == np.dtype(np.int32):
            return "i32"
        if nd == np.dtype(np.uint32):
            return "u32"
        if nd == np.dtype(np.bool_):
            return "bool"

        # WGSL (WebGPU) currently only supports a limited set of scalar types.
        raise LoopyTypeError(f"numpy dtype {nd} is not supported by WGSLTarget")


# }} }


# {{{ tiny WGSL AST


def _indent_lines(code: str, prefix: str) -> str:
    if not code:
        return ""

    return "\n".join(
        (prefix + line) if line else ""
        for line in code.splitlines())


@dataclass(frozen=True)
class WGSLRaw:
    code: str

    @override
    def __str__(self) -> str:  # noqa: D105
        return self.code


@dataclass(frozen=True)
class WGSLStatement:
    code: str

    @override
    def __str__(self) -> str:  # noqa: D105
        code = self.code.rstrip()
        if not code:
            return ";"

        # If there's an end-of-line comment, the semicolon must appear *before*
        # the comment. Otherwise WGSL parsers (correctly) report a missing ';'
        # at the next token.
        cmt_idx = code.find("//")
        if cmt_idx != -1:
            pre = code[:cmt_idx].rstrip()
            cmt = code[cmt_idx:].lstrip()
            if pre.endswith(";"):
                return f"{pre} {cmt}".rstrip()
            return f"{pre}; {cmt}".rstrip()

        if code.endswith(";"):
            return code
        return code + ";"


@dataclass(frozen=True)
class WGSLStatements:
    """A sequence of statements (no braces)."""

    contents: list[object]

    def __post_init__(self) -> None:
        # loopy.codegen may treat the block container type as an "AST list" and
        # (depending on the code path) may pass an existing Collection instance
        # into the Collection constructor. Coerce to a list defensively so that
        # __str__ and downstream code can rely on `contents` being a list.
        #
        # This intentionally allows nested WGSLStatements to exist as a single
        # element (i.e. not flattened), since flattening decisions belong to the
        # codegen pipeline, not the stringifier.
        if self.contents is None:
            object.__setattr__(self, "contents", [])
            return

        if isinstance(self.contents, WGSLStatements):
            object.__setattr__(self, "contents", [self.contents])
            return

        if not isinstance(self.contents, list):
            # Avoid turning a string into a list of characters.
            if isinstance(self.contents, str):
                object.__setattr__(self, "contents", [self.contents])
            else:
                object.__setattr__(self, "contents", list(self.contents))

    @override
    def __str__(self) -> str:  # noqa: D105
        return "\n".join(str(item) for item in self.contents if item is not None)


@dataclass(frozen=True)
class WGSLStatementsScope(WGSLStatements):
    """A statement sequence that must *not* be flattened away."""


@dataclass(frozen=True)
class WGSLBlock:
    """A braced WGSL block."""

    body: object

    @override
    def __str__(self) -> str:  # noqa: D105
        body_s = str(self.body)
        if not body_s.strip():
            return "{}"

        return "{\n" + _indent_lines(body_s, "  ") + "\n}"


@dataclass(frozen=True)
class WGSLIf:
    condition: str
    body: WGSLBlock

    @override
    def __str__(self) -> str:  # noqa: D105
        return f"if ({self.condition}) {self.body}"


@dataclass(frozen=True)
class WGSLFor:
    initializer: str
    condition: str
    continuing: str
    body: WGSLBlock

    @override
    def __str__(self) -> str:  # noqa: D105
        return (
            f"for ({self.initializer}; {self.condition}; {self.continuing}) "
            f"{self.body}")


@dataclass(frozen=True)
class WGSLFunction:
    decorators: tuple[str, ...]
    name: str
    parameters: tuple[str, ...]
    body: WGSLBlock
    return_type: str | None = None

    @override
    def __str__(self) -> str:  # noqa: D105
        deco = "\n".join(self.decorators)
        params = ", ".join(self.parameters)
        sig = f"fn {self.name}({params})"
        if self.return_type is not None:
            sig += f" -> {self.return_type}"

        if deco:
            return f"{deco}\n{sig} {self.body}"
        return f"{sig} {self.body}"


# }} }


# {{{ expression mappers


class WGSLExpressionToCodeMapper(CExpressionToCodeMapper):
    @override
    def map_if(self, expr: p.If, enclosing_prec: int) -> str:
        # WGSL has no ternary operator; use select(false, true, cond).
        from pymbolic.mapper.stringifier import PREC_CALL

        cond_ = self.rec(expr.condition, PREC_NONE)
        then_ = self.rec(expr.then, PREC_NONE)
        else_ = self.rec(expr.else_, PREC_NONE)

        return self.parenthesize_if_needed(
            f"select({else_}, {then_}, {cond_})",
            enclosing_prec, PREC_CALL)

    @override
    def map_array_literal(self, expr: object, enclosing_prec: int) -> str:
        # ArrayLiteral is currently only produced for initializers.
        raise LoopyError(
            "WGSLTarget does not currently support array literal initializers")


class ExpressionToWGSLExpressionMapper(ExpressionToCExpressionMapper):
    _GRID_AXES = "xyz"

    @override
    def map_sum(self, expr: p.Sum, type_context: TypeContext):
        # C relies on implicit int<->float conversion in arithmetic. WGSL does not.
        # Cast children to the inferred expression type (when needed) to avoid
        # emitting invalid mixed-type arithmetic (e.g. f32 + i32).
        result_dtype = self.infer_type(expr)
        inner_tc = dtype_to_type_context(self.kernel.target, result_dtype)
        return type(expr)(
            tuple(self.rec(ch, inner_tc, result_dtype) for ch in expr.children))

    @override
    def map_product(self, expr: p.Product, type_context: TypeContext):
        # See map_sum.
        result_dtype = self.infer_type(expr)
        inner_tc = dtype_to_type_context(self.kernel.target, result_dtype)
        return type(expr)(
            tuple(self.rec(ch, inner_tc, result_dtype) for ch in expr.children))

    @override
    def wrap_in_typecast(self,
                         actual_type: LoopyType,
                         needed_type: LoopyType,
                         s: Expression) -> Expression:
        if actual_type == needed_type:
            return s

        # WGSL does not allow implicit numeric<->bool conversions.
        if (
                isinstance(needed_type, NumpyType)
                and needed_type.numpy_dtype == np.dtype(np.bool_)
                and isinstance(actual_type, NumpyType)
                and actual_type.numpy_dtype.kind in "ifu"):
            return p.Comparison(s, "!=", 0)

        if (
                isinstance(actual_type, NumpyType)
                and actual_type.numpy_dtype == np.dtype(np.bool_)
                and isinstance(needed_type, NumpyType)
                and needed_type.numpy_dtype.kind in "ifu"):
            # WGSL doesn't allow bool->numeric casts. Use select/cast instead.
            registry = self.codegen_state.ast_builder.target.get_dtype_registry()
            cast_tp = registry.dtype_to_ctype(needed_type)
            return p.If(s, var(cast_tp)(1), var(cast_tp)(0))

        registry = self.codegen_state.ast_builder.target.get_dtype_registry()
        cast_tp = registry.dtype_to_ctype(needed_type)
        return var(cast_tp)(s)

    @override
    def map_constant(self, expr: object, type_context: TypeContext):
        from loopy.symbolic import Literal

        if isinstance(expr, complex):
            raise LoopyTypeError("WGSLTarget does not support complex constants")

        if isinstance(expr, np.generic):
            # Explicitly typed.
            # Avoid re-wrapping dtype() to keep type checkers happy.
            nd = expr.dtype

            if nd.kind == "c":
                raise LoopyTypeError("WGSLTarget does not support complex constants")

            if nd == np.dtype(np.float32):
                return Literal(repr(float(expr.item())))
            if nd == np.dtype(np.bool_):
                return Literal("true") if bool(expr.item()) else Literal("false")

            if nd.kind in "iu":
                bits = nd.itemsize * 8
                if bits > 32:
                    raise LoopyTypeError(
                        f"WGSLTarget does not support {bits}-bit integers")

                suffix = "u" if nd.kind == "u" else ""
                return Literal(repr(int(expr.item())) + suffix)

            raise LoopyTypeError(
                f"do not know how to generate WGSL for numpy dtype '{nd}'")

        if isinstance(expr, bool):
            return Literal("true") if expr else Literal("false")

        if isinstance(expr, int):
            if type_context == "f":
                return Literal(repr(float(expr)))
            return int(expr)

        if isinstance(expr, float):
            if not math.isfinite(expr):
                raise LoopyTypeError(f"WGSLTarget does not support {expr!r}")
            return Literal(repr(expr))

        raise LoopyTypeError(f"unsupported WGSL constant: {expr!r}")

    @override
    def map_variable(self, expr: p.Variable, type_context: TypeContext):
        if expr.name in self.codegen_state.var_subst_map:
            return self.rec(self.codegen_state.var_subst_map[expr.name], type_context)

        if expr.name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[expr.name]

            # ValueArgs live in a single uniform struct.
            if isinstance(arg, ValueArg):
                # Keep ValueArgs as ordinary variables in expressions so that
                # access analysis (affine index simplification) can still "see"
                # parameters like 'n'. The entry point prolog aliases these from
                # the uniform struct (see WGSLASTBuilder.generate_top_of_body).
                return var(expr.name)

            # Scalar arrays are represented as a storage buffer and accessed at 0.
            from loopy.kernel.array import ArrayBase
            if isinstance(arg, ArrayBase) and arg.shape == ():
                from loopy.kernel.array import _apply_offset
                from loopy.symbolic import simplify_using_aff

                subscript = _apply_offset(0, arg)
                return self.make_subscript(
                    arg,
                    var(expr.name),
                    simplify_using_aff(
                        self.kernel,
                        self.rec_arith(cast(ArithmeticExpression, subscript), "i"),
                    ),
                )

        return super().map_variable(expr, type_context)

    @override
    def make_subscript(self, array: object, base_expr: Expression, subscript: Expression):
        # Kernel arguments are declared as structs with a runtime-sized "data" array.
        if isinstance(array, (ArrayArg, ConstantArg)):
            base_expr = p.Lookup(base_expr, "data")

        if isinstance(array, TemporaryVariable) and array.address_space == AddressSpace.GLOBAL:
            base_expr = p.Lookup(base_expr, "data")

        return super().make_subscript(array, base_expr, subscript)

    def map_group_hw_index(
            self,
            expr: GroupHardwareAxisIndex,
            type_context: TypeContext,
        ):
        ax = self._GRID_AXES[expr.axis]
        idx_dtype = self.kernel.index_dtype
        cast_tp = self.codegen_state.ast_builder.target.get_dtype_registry().dtype_to_ctype(idx_dtype)
        return var(cast_tp)(p.Lookup(var(WGSLASTBuilder.workgroup_id_arg_name), ax))

    def map_local_hw_index(
            self,
            expr: LocalHardwareAxisIndex,
            type_context: TypeContext,
        ):
        ax = self._GRID_AXES[expr.axis]
        idx_dtype = self.kernel.index_dtype
        cast_tp = self.codegen_state.ast_builder.target.get_dtype_registry().dtype_to_ctype(idx_dtype)
        return var(cast_tp)(p.Lookup(var(WGSLASTBuilder.local_invocation_id_arg_name), ax))


# }} }


# {{{ callables


class WGSLBuiltinCallable(ScalarCallable):
    """A minimal scalar callable for WGSL builtins."""

    @override
    def with_types(self,
                   arg_id_to_dtype: Mapping[int | str, LoopyType],
                   clbl_inf_ctx: CallablesInferenceContext,
                   ) -> tuple[WGSLBuiltinCallable, CallablesInferenceContext]:
        name = self.name

        for id_ in arg_id_to_dtype:
            if not isinstance(id_, int):
                raise LoopyError(f"'{name}' can take only positional arguments")

        arg_num_to_dtype = {cast(int, id_): t for id_, t in arg_id_to_dtype.items()}

        def need_args(nargs: int):
            # allow missing types during inference
            if any(i not in arg_num_to_dtype or arg_num_to_dtype[i] is None
                   for i in range(nargs)):
                return False
            return True

        def as_numpy_dtype(dtype: LoopyType) -> np.dtype[np.generic]:
            if not isinstance(dtype, NumpyType):
                raise LoopyTypeError(f"WGSLTarget only supports numpy dtypes, got {dtype}")
            return dtype.numpy_dtype

        if name in {"min", "max"}:
            if not need_args(2):
                return (self.copy(arg_id_to_dtype=arg_num_to_dtype), clbl_inf_ctx)

            common = np.result_type(
                as_numpy_dtype(arg_num_to_dtype[0]),
                as_numpy_dtype(arg_num_to_dtype[1]))

            if common not in {np.dtype(np.int32), np.dtype(np.uint32), np.dtype(np.float32)}:
                raise LoopyTypeError(f"'{name}' does not support dtype {common}")

            lpy = NumpyType(common)
            return (
                self.copy(
                    name_in_target=name,
                    arg_id_to_dtype={-1: lpy, 0: lpy, 1: lpy},
                ),
                clbl_inf_ctx,
            )

        if name == "clamp":
            if not need_args(3):
                return (self.copy(arg_id_to_dtype=arg_num_to_dtype), clbl_inf_ctx)

            common = np.result_type(
                as_numpy_dtype(arg_num_to_dtype[0]),
                as_numpy_dtype(arg_num_to_dtype[1]),
                as_numpy_dtype(arg_num_to_dtype[2]),
            )
            if common not in {np.dtype(np.int32), np.dtype(np.uint32), np.dtype(np.float32)}:
                raise LoopyTypeError(f"'{name}' does not support dtype {common}")

            lpy = NumpyType(common)
            return (
                self.copy(
                    name_in_target=name,
                    arg_id_to_dtype={-1: lpy, 0: lpy, 1: lpy, 2: lpy},
                ),
                clbl_inf_ctx,
            )

        if name in {"abs"}:
            if not need_args(1):
                return (self.copy(arg_id_to_dtype=arg_num_to_dtype), clbl_inf_ctx)

            dtype = as_numpy_dtype(arg_num_to_dtype[0])
            if dtype not in {np.dtype(np.int32), np.dtype(np.float32)}:
                raise LoopyTypeError(f"'{name}' does not support dtype {dtype}")

            lpy = NumpyType(dtype)
            return (
                self.copy(name_in_target=name, arg_id_to_dtype={-1: lpy, 0: lpy}),
                clbl_inf_ctx,
            )

        if name in {"sqrt", "sin", "cos", "tan", "exp", "log", "log2"}:
            if not need_args(1):
                return (self.copy(arg_id_to_dtype=arg_num_to_dtype), clbl_inf_ctx)

            # Accept integer input by casting to f32.
            lpy = NumpyType(np.dtype(np.float32))
            return (
                self.copy(name_in_target=name, arg_id_to_dtype={-1: lpy, 0: lpy}),
                clbl_inf_ctx,
            )

        if name == "pow":
            if not need_args(2):
                return (self.copy(arg_id_to_dtype=arg_num_to_dtype), clbl_inf_ctx)

            lpy = NumpyType(np.dtype(np.float32))
            return (
                self.copy(name_in_target=name, arg_id_to_dtype={-1: lpy, 0: lpy, 1: lpy}),
                clbl_inf_ctx,
            )

        raise LoopyError(f"WGSLTarget does not know builtin callable '{name}'")


def get_wgsl_callables() -> dict[str, ScalarCallable]:
    # Keep this small and WGSL-accurate (no fmin/fmax/etc).
    func_ids = {
        "abs",
        "clamp",
        "cos",
        "exp",
        "log",
        "log2",
        "max",
        "min",
        "pow",
        "sin",
        "sqrt",
        "tan",
    }
    return {id_: WGSLBuiltinCallable(name=id_) for id_ in func_ids}


# }} }


# {{{ preamble generator


def _wgsl_decl_for_params(kernel: LoopKernel) -> str:
    value_args = [arg for arg in kernel.args if isinstance(arg, ValueArg)]
    if not value_args:
        return ""

    reg = kernel.target.get_dtype_registry()
    fields = []
    for arg in value_args:
        assert arg.dtype is not None
        wgsl_tp = reg.dtype_to_ctype(arg.dtype)
        if wgsl_tp == "bool":
            # bool is not a host-shareable type in WebGPU buffer address spaces.
            raise LoopyTypeError(
                "WGSLTarget does not support bool ValueArgs in uniform buffers")

        fields.append(f"  {arg.name}: {wgsl_tp},")

    fields_s = "\n".join(fields)
    return (
        "struct LoopyParams {\n"
        f"{fields_s}\n"
        "};\n"
        f"@group(0) @binding(0) var<uniform> {WGSLASTBuilder.params_var_name}: LoopyParams;\n"
    )


def _wgsl_decl_for_buffers(kernel: LoopKernel, start_binding: int) -> str:
    written = kernel.get_written_variables()
    reg = kernel.target.get_dtype_registry()

    decls = []
    binding = start_binding

    for arg in kernel.args:
        if isinstance(arg, ValueArg):
            continue

        if isinstance(arg, (ArrayArg, ConstantArg)):
            assert arg.dtype is not None
            el_tp = reg.dtype_to_ctype(arg.dtype)
            if el_tp == "bool":
                # bool is not a host-shareable type in WebGPU buffer address spaces.
                raise LoopyTypeError(
                    "WGSLTarget does not support bool buffers")

            struct_name = f"LoopyBuf_{arg.name}"
            access = "read_write" if arg.name in written else "read"
            if isinstance(arg, ConstantArg):
                access = "read"

            decls.append(
                "struct {sn} {{\n  data: array<{tp}>,\n}};\n"
                "@group(0) @binding({b}) var<storage, {acc}> {name}: {sn};\n".format(
                    sn=struct_name,
                    tp=el_tp,
                    b=binding,
                    acc=access,
                    name=arg.name,
                )
            )
            binding += 1
            continue

        raise LoopyError(f"WGSLTarget does not support kernel argument '{arg}'")

    return "\n".join(decls)


def _wgsl_decl_for_workgroup_temporaries(kernel: LoopKernel) -> str:
    """Generate module-scope `var<workgroup>` declarations for LOCAL temporaries.

    WGSL workgroup variables must be module-scope, so they cannot be emitted via
    :meth:`WGSLASTBuilder.get_temporary_decls` (which generates function-scope
    declarations).
    """
    reg = kernel.target.get_dtype_registry()

    decls: list[str] = []
    for tv_name in sorted(kernel.temporary_variables):
        tv = kernel.temporary_variables[tv_name]

        if tv.address_space != AddressSpace.LOCAL:
            continue

        if tv.initializer is not None:
            raise LoopyError(
                "WGSLTarget does not currently support LOCAL temporary "
                "initializers")
        if tv.base_storage:
            raise LoopyError(
                "WGSLTarget does not currently support base_storage LOCAL "
                "temporaries")

        assert tv.dtype is not None
        tp = reg.dtype_to_ctype(tv.dtype)

        shape = tv.storage_shape if tv.storage_shape else tv.shape
        from loopy.typing import auto
        if shape is auto:
            raise LoopyError(
                f"WGSLTarget requires a concrete shape for '{tv.name}'")

        if shape in (None, ()):
            decls.append(f"var<workgroup> {tv.name}: {tp};")
        else:
            if not isinstance(shape, tuple):
                raise LoopyError(
                    f"WGSLTarget expected a tuple shape for '{tv.name}', got {shape!r}")

            if not all(isinstance(ax, int) for ax in shape):
                raise LoopyError(
                    "WGSLTarget requires constant-size workgroup arrays for "
                    f"'{tv.name}'")

            size = 1
            for ax in shape:
                size *= int(ax)

            decls.append(f"var<workgroup> {tv.name}: array<{tp}, {size}>;")

    return "\n".join(decls)


def wgsl_preamble_generator(preamble_info: PreambleInfo) -> Iterator[tuple[str, str]]:
    kernel = preamble_info.kernel

    # Basic type sanity early, to fail with a clear message.
    for arg in kernel.args:
        if arg.dtype is None:
            continue
        if not isinstance(arg.dtype, NumpyType):
            raise LoopyTypeError(f"WGSLTarget only supports numpy dtypes, got {arg.dtype}")

    value_args = [arg for arg in kernel.args if isinstance(arg, ValueArg)]
    has_params = bool(value_args)

    parts = [
        "// Generated by loopy (WGSLTarget)\n",
    ]

    # Uniform params (binding 0) if needed.
    params_part = _wgsl_decl_for_params(kernel)
    if params_part:
        parts.append(params_part)

    start_binding = 1 if has_params else 0
    buffers_part = _wgsl_decl_for_buffers(kernel, start_binding)
    if buffers_part:
        parts.append(buffers_part)

    workgroup_temps_part = _wgsl_decl_for_workgroup_temporaries(kernel)
    if workgroup_temps_part:
        parts.append(workgroup_temps_part)

    yield ("00_wgsl_bindings", "\n".join(p.strip("\n") for p in parts if p))

def wgsl_int_math_preamble_generator(preamble_info: PreambleInfo) -> Iterator[tuple[str, str]]:
    """Emit WGSL helpers for floor division and modulo used by loopy's codegen.

    Loopy's index expression generation reuses CTarget naming for these late-bound
    helpers (via :class:`~loopy.codegen.SeenFunction`), e.g.

      - loopy_floor_div_pos_b_int32
      - loopy_mod_int32

    WGSL does not provide a built-in floor division operator for signed integers
    (i32 division truncates toward zero), so we provide the same semantics as the
    CTarget helpers.
    """
    c_funcs = {func.c_name for func in preamble_info.seen_functions}

    needed = {
        name for name in c_funcs
        if name.startswith("loopy_floor_div") or name.startswith("loopy_mod")
    }
    if not needed:
        return

    # Emit a small, fixed set of helpers for the scalar integer types WGSLTarget supports.
    # (loopy currently only supports int32/uint32 scalars for WGSLTarget anyway.)
    parts: list[str] = []
    parts.append("// Integer math helpers (floor div / mod) for loopy\n")

    if (
            "loopy_floor_div_int32" in needed
            or "loopy_floor_div_pos_b_int32" in needed
            or "loopy_mod_int32" in needed
            or "loopy_mod_pos_b_int32" in needed
    ):
        parts.append(
            "fn loopy_floor_div_int32(a_in: i32, b_in: i32) -> i32 {\n"
            "  var a: i32 = a_in;\n"
            "  let b: i32 = b_in;\n"
            "  // If signs differ, adjust to ensure floor semantics.\n"
            "  if ((a < 0) != (b < 0)) {\n"
            "    let b_lt0: i32 = select(0, 1, b < 0);\n"
            "    let b_ge0: i32 = select(0, 1, b >= 0);\n"
            "    a = a - (b + b_lt0 - b_ge0);\n"
            "  }\n"
            "  return a / b;\n"
            "}\n"
        )

        parts.append(
            "fn loopy_floor_div_pos_b_int32(a_in: i32, b_in: i32) -> i32 {\n"
            "  var a: i32 = a_in;\n"
            "  let b: i32 = b_in;\n"
            "  // b is assumed positive.\n"
            "  if (a < 0) {\n"
            "    a = a - (b - 1);\n"
            "  }\n"
            "  return a / b;\n"
            "}\n"
        )

        parts.append(
            "fn loopy_mod_int32(a_in: i32, b_in: i32) -> i32 {\n"
            "  let a: i32 = a_in;\n"
            "  let b: i32 = b_in;\n"
            "  var result: i32 = a % b;\n"
            "  if (result < 0 && b > 0) {\n"
            "    result = result + b;\n"
            "  }\n"
            "  if (result > 0 && b < 0) {\n"
            "    result = result + b;\n"
            "  }\n"
            "  return result;\n"
            "}\n"
        )

        parts.append(
            "fn loopy_mod_pos_b_int32(a_in: i32, b_in: i32) -> i32 {\n"
            "  let a: i32 = a_in;\n"
            "  let b: i32 = b_in;\n"
            "  // b is assumed positive.\n"
            "  var result: i32 = a % b;\n"
            "  if (result < 0) {\n"
            "    result = result + b;\n"
            "  }\n"
            "  return result;\n"
            "}\n"
        )

    if (
            "loopy_floor_div_uint32" in needed
            or "loopy_floor_div_pos_b_uint32" in needed
            or "loopy_mod_uint32" in needed
            or "loopy_mod_pos_b_uint32" in needed
    ):
        parts.append(
            "fn loopy_floor_div_uint32(a: u32, b: u32) -> u32 { return a / b; }\n"
            "fn loopy_floor_div_pos_b_uint32(a: u32, b: u32) -> u32 { return a / b; }\n"
            "fn loopy_mod_uint32(a: u32, b: u32) -> u32 { return a % b; }\n"
            "fn loopy_mod_pos_b_uint32(a: u32, b: u32) -> u32 { return a % b; }\n"
        )

    yield ("01_wgsl_int_math", "\n".join(p.strip("\n") for p in parts if p))


# }} }


# {{{ target + ast builder


class WGSLTarget(TargetBase):
    """A code generation target for WGSL compute shaders."""

    @override
    def split_kernel_at_global_barriers(self) -> bool:
        # WebGPU/WGSL has no cross-workgroup global barrier.
        return True

    @override
    def get_host_ast_builder(self):
        # DummyHostASTBuilder is parametrized as ASTBuilderBase[None]. Cast to
        # satisfy invariance of ASTBuilderBase in type checkers.
        return cast(ASTBuilderBase[Any], DummyHostASTBuilder(self))

    @override
    def get_device_ast_builder(self):
        return cast(ASTBuilderBase[Any], WGSLASTBuilder(self))

    @memoize_method
    def get_dtype_registry(self) -> DTypeRegistry:
        return WGSLDTypeRegistry()

    @override
    def is_vector_dtype(self, dtype: LoopyType) -> bool:
        return False

    @override
    def vector_dtype(self, base: LoopyType, count: int) -> LoopyType:
        raise LoopyTypeError("WGSLTarget does not currently support vector dtypes")

    def get_kernel_executor_cache_key(self, *args: Any, **kwargs: Any):
        # No execution support at the moment, but TranslationUnit.executor()
        # expects this to exist.
        return (type(self),)

    def get_kernel_executor(  # type: ignore[override]
            self,
            t_unit: TranslationUnit,
            *args: Any,
            entrypoint: str,
            **kwargs: Any,
            ):
        raise LoopyError(
            "WGSLTarget does not currently implement execution. "
            "Use generate_code_v2(...).device_code() to obtain WGSL and "
            "dispatch it via WebGPU (browser/JS), wgpu-native, etc.")


@dataclass(frozen=True)
class WGSLASTBuilder(ASTBuilderBase[object]):
    target: WGSLTarget

    # These identifiers are part of the generated shader interface.
    params_var_name = "_lpy_params"
    workgroup_id_arg_name = "_lpy_workgroup_id"
    local_invocation_id_arg_name = "_lpy_local_invocation_id"

    # {{{ library

    @property
    @override
    def known_callables(self):
        callables = super().known_callables
        callables.update(get_wgsl_callables())
        return callables

    @override
    def preamble_generators(self):
        return [
            *super().preamble_generators(),
            wgsl_preamble_generator,
            wgsl_int_math_preamble_generator,
        ]

    # }} }

    # {{{ code generation guts

    @property
    @override
    def ast_module(self):
        # Used by loopy.codegen to stitch together device program ASTs.
        class _WGSLASTModule:
            Collection = WGSLStatements

        return _WGSLASTModule

    def get_c_expression_to_code_mapper(self) -> WGSLExpressionToCodeMapper:
        return WGSLExpressionToCodeMapper()

    @override
    def get_expression_to_code_mapper(self, codegen_state: CodeGenerationState):
        return ExpressionToWGSLExpressionMapper(codegen_state)

    @property
    @override
    def ast_block_class(self):
        return WGSLStatements

    @property
    @override
    def ast_block_scope_class(self):
        return WGSLStatementsScope

    @override
    def get_function_declaration(
            self,
            codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult[object],
            schedule_index: int,
            ) -> tuple[Sequence[tuple[str, str]], object | None]:
        # WGSL uses module-scope bindings; entry points only take builtins.
        return [], None

    @override
    def generate_top_of_body(self, codegen_state: CodeGenerationState) -> Sequence[object]:
        """Emit a small prolog for entry points.

        In WebGPU/WGSL, scalar kernel parameters are passed in via a uniform
        buffer. To keep index expressions affine (important for access analysis
        and simplification), we keep ValueArgs as variables in expressions and
        alias them from the uniform struct here.
        """
        kernel = codegen_state.kernel
        value_args = [arg for arg in kernel.args if isinstance(arg, ValueArg)]
        if not value_args:
            return []

        reg = kernel.target.get_dtype_registry()
        decls: list[object] = []

        for arg in value_args:
            assert arg.dtype is not None
            wgsl_tp = reg.dtype_to_ctype(arg.dtype)
            if wgsl_tp == "bool":
                # bool is not host-shareable in uniform buffers, and we already
                # reject these in the preamble generator. Keep the message local.
                raise LoopyTypeError(
                    "WGSLTarget does not support bool ValueArgs in uniform buffers")

            decls.append(
                WGSLStatement(
                    f"let {arg.name}: {wgsl_tp} = {self.params_var_name}.{arg.name}"
                )
            )

        return decls

    @override
    def get_function_definition(
            self,
            codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult[object],
            schedule_index: int,
            function_decl: object | None,
            function_body: object,
            ) -> object:
        kernel = codegen_state.kernel

        from loopy.schedule import get_insn_ids_for_block_at

        assert kernel.linearization is not None
        insn_ids = get_insn_ids_for_block_at(kernel.linearization, schedule_index)
        _gsize, lsize = kernel.get_grid_sizes_for_insn_ids_as_exprs(
            insn_ids, codegen_state.callables_table)

        # @workgroup_size must be a compile-time constant in WGSL.
        if not lsize:
            wg_size = (1, 1, 1)
        else:
            if len(lsize) > 3:
                raise LoopyError("WGSLTarget supports at most 3 local axes")

            from loopy.symbolic import get_dependencies
            if get_dependencies(lsize):
                raise LoopyError(
                    "WGSLTarget requires constant workgroup_size; "
                    "please fix parameters or choose constant local sizes")

            wg_size = tuple(int(cast(int, ax)) for ax in (*lsize, *(3-len(lsize))*(1,)))
            wg_size = cast(tuple[int, int, int], wg_size)

        wg_size_str = ", ".join(str(s) for s in wg_size)

        name = codegen_state.gen_program_name

        params = (
            f"@builtin(workgroup_id) {self.workgroup_id_arg_name}: vec3<u32>",
            f"@builtin(local_invocation_id) {self.local_invocation_id_arg_name}: vec3<u32>",
        )

        if not isinstance(function_body, WGSLStatements):
            function_body = WGSLStatements([function_body])

        return WGSLFunction(
            decorators=(
                "@compute",
                f"@workgroup_size({wg_size_str})",
            ),
            name=name,
            parameters=params,
            body=WGSLBlock(function_body),
        )

    @override
    def get_temporary_decls(self, codegen_state: CodeGenerationState, schedule_index: int):
        from loopy.schedule.tools import (
            supporting_temporary_names,
            temporaries_read_in_subkernel,
            temporaries_written_in_subkernel,
        )

        kernel = codegen_state.kernel
        assert kernel.linearization is not None
        from loopy.schedule import CallKernel
        subkernel_name = cast(CallKernel, kernel.linearization[schedule_index]).kernel_name

        sub_knl_temps = (
            temporaries_read_in_subkernel(kernel, subkernel_name)
            | temporaries_written_in_subkernel(kernel, subkernel_name)
        )
        sub_knl_temps |= supporting_temporary_names(kernel, sub_knl_temps)

        if not sub_knl_temps:
            return []

        reg = self.target.get_dtype_registry()

        decls: list[object] = []
        for tv_name in sorted(sub_knl_temps):
            tv = kernel.temporary_variables[tv_name]

            if tv.initializer is not None:
                raise LoopyError(
                    "WGSLTarget does not currently support temporary initializers")
            if tv.base_storage:
                raise LoopyError(
                    "WGSLTarget does not currently support base_storage temporaries")
            if tv.address_space == AddressSpace.LOCAL:
                # Workgroup temporaries are declared at module scope by
                # :func:`wgsl_preamble_generator`.
                continue
            if tv.address_space == AddressSpace.GLOBAL:
                raise LoopyError(
                    "WGSLTarget does not currently support GLOBAL temporaries; "
                    "pass them as explicit storage buffers")

            assert tv.dtype is not None
            tp = reg.dtype_to_ctype(tv.dtype)

            shape = tv.storage_shape if tv.storage_shape else tv.shape
            from loopy.typing import auto
            if shape is auto:
                raise LoopyError(
                    f"WGSLTarget requires a concrete shape for '{tv.name}'")

            if shape in (None, ()):
                decls.append(WGSLStatement(f"var {tv.name}: {tp}"))
            else:
                if not isinstance(shape, tuple):
                    raise LoopyError(
                        f"WGSLTarget expected a tuple shape for '{tv.name}', got {shape!r}")

                if not all(isinstance(ax, int) for ax in shape):
                    raise LoopyError(
                        f"WGSLTarget requires constant-size private arrays for '{tv.name}'")

                size = 1
                for ax in shape:
                    size *= int(ax)

                decls.append(WGSLStatement(f"var {tv.name}: array<{tp}, {size}>"))

        if decls:
            decls.append(WGSLRaw(""))
            return decls

        return []

    @override
    def get_kernel_call(self, codegen_state, subkernel_name, gsize, lsize):
        return None

    @override
    def add_vector_access(self, access_expr, index: int):
        axes = "xyzw"
        return p.Lookup(access_expr, axes[index])

    @override
    def emit_barrier(self, synchronization_kind: str, mem_kind: str, comment: str | None):
        if synchronization_kind != "local":
            raise LoopyError("WGSLTarget can only emit local barriers")

        cmt = f" // {comment}" if comment else ""
        if mem_kind == "local":
            return WGSLStatement(f"workgroupBarrier(){cmt}")
        if mem_kind == "global":
            # Storage barrier does not synchronize control flow, so include a
            # workgroupBarrier as well.
            return WGSLStatements([
                WGSLStatement("storageBarrier()"),
                WGSLStatement(f"workgroupBarrier(){cmt}"),
            ])

        raise LoopyError(f"unknown mem_kind '{mem_kind}'")

    @override
    def emit_assignment(self, codegen_state: CodeGenerationState, insn: Assignment):
        if insn.atomicity:
            raise LoopyError("WGSLTarget does not currently support atomic ops")

        ecm = codegen_state.expression_to_code_mapper
        lhs = str(ecm(insn.assignee, prec=PREC_NONE, type_context=None))
        rhs = str(ecm(insn.expression, prec=PREC_NONE, type_context=None))
        return WGSLStatement(f"{lhs} = {rhs}")

    @override
    def emit_multiple_assignment(self, codegen_state, insn):
        raise LoopyError("WGSLTarget does not currently support multiple assignment")

    @override
    def emit_sequential_loop(self,
                             codegen_state: CodeGenerationState,
                             iname: str,
                             iname_dtype: LoopyType,
                             lbound: Expression,
                             ubound: Expression,
                             inner: object,
                             hints: Sequence[object],
                             ):
        ecm = codegen_state.expression_to_code_mapper

        reg = self.target.get_dtype_registry()
        iname_tp = reg.dtype_to_ctype(iname_dtype)

        lbound_s = str(ecm(lbound, PREC_NONE, type_context="i"))
        ubound_s = str(ecm(ubound, PREC_NONE, type_context="i"))

        if not isinstance(inner, WGSLStatements):
            inner = WGSLStatements([inner])

        loop = WGSLFor(
            initializer=f"var {iname}: {iname_tp} = {lbound_s}",
            condition=f"{iname} <= {ubound_s}",
            continuing=f"{iname} = {iname} + 1",
            body=WGSLBlock(inner),
        )

        if hints:
            return WGSLStatements([*list(hints), loop])
        return loop

    @override
    def emit_unroll_hint(self, value: int | None):
        if value is None:
            return WGSLRaw("// unroll")
        return WGSLRaw(f"// unroll {value}")

    @property
    @override
    def can_implement_conditionals(self):
        return True

    @override
    def emit_if(self, condition_str: str, ast: object):
        if not isinstance(ast, WGSLStatements):
            ast = WGSLStatements([ast])
        return WGSLIf(str(condition_str), WGSLBlock(ast))

    @override
    def emit_initializer(self, codegen_state, dtype, name, val_str, is_const: bool):
        reg = self.target.get_dtype_registry()
        tp = reg.dtype_to_ctype(dtype)

        kw = "let" if is_const else "var"
        return WGSLStatement(f"{kw} {name}: {tp} = {val_str}")

    @override
    def emit_declaration_scope(self, codegen_state, inner):
        if not isinstance(inner, WGSLStatements):
            inner = WGSLStatements([inner])
        return WGSLBlock(inner)

    @override
    def emit_blank_line(self):
        return WGSLRaw("")

    @override
    def emit_comment(self, s: str):
        return WGSLRaw(f"// {s}")

    @override
    def emit_noop_with_comment(self, s: str):
        return WGSLRaw(f"// {s}")

    # }} }


# }} }


# {{{ WebGPU interface helpers


def _round_up(value: int, align: int) -> int:
    if align <= 0:
        raise ValueError(f"align must be positive, got {align}")
    return ((value + align - 1) // align) * align


@dataclass(frozen=True)
class WGSLUniformFieldLayout:
    name: str
    wgsl_type: str
    offset: int
    size: int
    align: int


@dataclass(frozen=True)
class WGSLParamsLayout:
    group: int
    binding: int
    struct_name: str
    var_name: str
    size: int
    align: int
    fields: tuple[WGSLUniformFieldLayout, ...]


@dataclass(frozen=True)
class WGSLStorageBindingLayout:
    group: int
    binding: int
    name: str
    struct_name: str
    access: str  # "read" or "read_write" (WGSL spelling)
    element_type: str  # WGSL scalar type
    element_nbytes: int

    @property
    def webgpu_buffer_binding_type(self) -> str:
        # WebGPU bind group layout types.
        # https://www.w3.org/TR/webgpu/#enumdef-gpubufferbindingtype
        return "read-only-storage" if self.access == "read" else "storage"


@dataclass(frozen=True)
class WGSLDispatchInfo:
    entrypoint: str
    workgroup_size: tuple[int, int, int]
    workgroups: tuple[Expression, Expression, Expression]


@dataclass(frozen=True)
class WGSLWebGPUInfo:
    """A minimal reflection helper for dispatching loopy-generated WGSL via WebGPU.

    This is intentionally lightweight and limited to information that is stable
    across WebGPU implementations: bind group layout and dispatch sizes.
    """

    entrypoint: str
    wgsl_source: str
    params: WGSLParamsLayout | None
    storage_bindings: tuple[WGSLStorageBindingLayout, ...]
    dispatches: tuple[WGSLDispatchInfo, ...]


def _wgsl_scalar_layout(wgsl_tp: str) -> tuple[int, int]:
    # WGSL scalar sizes/alignments (host-shareable subset).
    if wgsl_tp in {"f32", "i32", "u32"}:
        return (4, 4)

    # bool exists in WGSL, but it is not host-shareable for uniform/storage
    # buffers. It may still appear in private/workgroup memory.
    if wgsl_tp == "bool":
        return (4, 4)

    raise LoopyTypeError(f"WGSLTarget does not know scalar layout for '{wgsl_tp}'")


def _build_params_layout(kernel: LoopKernel) -> WGSLParamsLayout | None:
    value_args = [arg for arg in kernel.args if isinstance(arg, ValueArg)]
    if not value_args:
        return None

    reg = kernel.target.get_dtype_registry()

    fields: list[WGSLUniformFieldLayout] = []
    offset = 0
    max_align = 1

    for arg in value_args:
        assert arg.dtype is not None
        wgsl_tp = reg.dtype_to_ctype(arg.dtype)
        if wgsl_tp == "bool":
            raise LoopyTypeError(
                "WGSLTarget does not support bool ValueArgs in uniform buffers")

        align, size = _wgsl_scalar_layout(wgsl_tp)
        offset = _round_up(offset, align)
        fields.append(WGSLUniformFieldLayout(
            name=arg.name,
            wgsl_type=wgsl_tp,
            offset=offset,
            size=size,
            align=align,
        ))
        offset += size
        max_align = max(max_align, align)

    # WebGPU requires uniform buffer bindings to be aligned; rounding up to 16
    # keeps the host-side packing safe and portable.
    struct_align = max(16, max_align)
    struct_size = _round_up(offset, struct_align)

    return WGSLParamsLayout(
        group=0,
        binding=0,
        struct_name="LoopyParams",
        var_name=WGSLASTBuilder.params_var_name,
        size=struct_size,
        align=struct_align,
        fields=tuple(fields),
    )


def _build_storage_bindings(kernel: LoopKernel) -> tuple[WGSLStorageBindingLayout, ...]:
    written = kernel.get_written_variables()
    reg = kernel.target.get_dtype_registry()

    has_params = any(isinstance(arg, ValueArg) for arg in kernel.args)
    binding = 1 if has_params else 0

    result: list[WGSLStorageBindingLayout] = []
    for arg in kernel.args:
        if isinstance(arg, ValueArg):
            continue

        if isinstance(arg, (ArrayArg, ConstantArg)):
            assert arg.dtype is not None
            el_tp = reg.dtype_to_ctype(arg.dtype)
            if el_tp == "bool":
                raise LoopyTypeError("WGSLTarget does not support bool buffers")

            access = "read_write" if arg.name in written else "read"
            if isinstance(arg, ConstantArg):
                access = "read"

            if not isinstance(arg.dtype, NumpyType):
                raise LoopyTypeError(f"WGSLTarget only supports numpy dtypes, got {arg.dtype}")

            result.append(WGSLStorageBindingLayout(
                group=0,
                binding=binding,
                name=arg.name,
                struct_name=f"LoopyBuf_{arg.name}",
                access=access,
                element_type=el_tp,
                element_nbytes=arg.dtype.numpy_dtype.itemsize,
            ))
            binding += 1
            continue

        raise LoopyError(f"WGSLTarget does not support kernel argument '{arg}'")

    return tuple(result)


def _prepare_translation_unit_for_wgsl_info(
        t_unit: TranslationUnit,
        *,
        no_cache: bool,
        ) -> TranslationUnit:
    # This is similar to what :func:`loopy.codegen.generate_code_v2` does, but we
    # need access to the linearized kernel to compute dispatch sizes.
    from loopy.kernel import KernelState, LoopKernel
    from loopy.translation_unit import make_program

    if isinstance(t_unit, LoopKernel):
        t_unit = make_program(t_unit)

    if not isinstance(t_unit.target, WGSLTarget):
        raise LoopyError(
            f"WGSL WebGPU info requires WGSLTarget; got target "
            f"{type(t_unit.target).__name__}")

    from loopy.preprocess import preprocess_program
    from contextlib import nullcontext
    from loopy import CacheMode
    # This helper should be usable in restricted environments (CI sandboxes,
    # Pyodide, etc). When *no_cache* is requested, disable all of loopy's
    # persistent caches for the duration of these transformations.
    with CacheMode(False) if no_cache else nullcontext():
        if t_unit.state < KernelState.PREPROCESSED:
            if no_cache:
                t_unit = preprocess_program(t_unit, _no_memoize_on_disk=True)
            else:
                t_unit = preprocess_program(t_unit)

        from loopy.type_inference import infer_unknown_types
        t_unit = infer_unknown_types(t_unit, expect_completion=True)

        if t_unit.state < KernelState.LINEARIZED:
            from loopy.schedule import linearize
            t_unit = linearize(t_unit)

    return t_unit


def get_wgsl_webgpu_info(
        t_unit: TranslationUnit,
        *,
        entrypoint: str | None = None,
        no_cache: bool = True,
        ) -> WGSLWebGPUInfo:
    """Return WGSL source + minimal WebGPU reflection for a translation unit.

    :arg no_cache: If *True* (default), disables loopy's on-disk caches for the
        preprocessing/codegen steps performed here. This makes the helper usable
        in restricted environments (e.g. sandboxed CI, Pyodide).
    """
    t_unit = _prepare_translation_unit_for_wgsl_info(t_unit, no_cache=no_cache)

    if entrypoint is None:
        if len(t_unit.entrypoints) != 1:
            raise ValueError(
                "TranslationUnit has multiple possible entrypoints; "
                "specify 'entrypoint'.")
        entrypoint, = t_unit.entrypoints
    else:
        if entrypoint not in t_unit.entrypoints:
            raise LoopyError(
                f"'{entrypoint}' is not an entrypoint of this TranslationUnit")

    from loopy.kernel.function_interface import CallableKernel
    clbl = t_unit.callables_table[entrypoint]
    if not isinstance(clbl, CallableKernel):
        raise LoopyError(f"entrypoint '{entrypoint}' is not a CallableKernel")

    kernel = clbl.subkernel
    assert kernel.linearization is not None

    params_layout = _build_params_layout(kernel)
    storage_bindings = _build_storage_bindings(kernel)

    # Collect dispatch info for each subkernel (CallKernel schedule item).
    from loopy.schedule import CallKernel, get_insn_ids_for_block_at
    from loopy.symbolic import get_dependencies

    dispatches: list[WGSLDispatchInfo] = []
    for sched_idx, sched_item in enumerate(kernel.linearization):
        if not isinstance(sched_item, CallKernel):
            continue

        insn_ids = get_insn_ids_for_block_at(kernel.linearization, sched_idx)
        gsize, lsize = kernel.get_grid_sizes_for_insn_ids_as_exprs(
            insn_ids, t_unit.callables_table)

        if not lsize:
            wg_size = (1, 1, 1)
        else:
            if len(lsize) > 3:
                raise LoopyError("WGSLTarget supports at most 3 local axes")
            if get_dependencies(lsize):
                raise LoopyError(
                    "WGSLTarget requires constant workgroup_size; "
                    "please choose constant local sizes")
            wg_size = tuple(int(cast(int, ax)) for ax in (*lsize, *(3-len(lsize))*(1,)))
            wg_size = cast(tuple[int, int, int], wg_size)

        # dispatchWorkgroups expects workgroup counts, which correspond to loopy's
        # GroupInameTag axes sizes ("gsize" here).
        g0 = gsize[0] if len(gsize) > 0 else 1
        g1 = gsize[1] if len(gsize) > 1 else 1
        g2 = gsize[2] if len(gsize) > 2 else 1

        dispatches.append(WGSLDispatchInfo(
            entrypoint=sched_item.kernel_name,
            workgroup_size=wg_size,
            workgroups=(g0, g1, g2),
        ))

    if not dispatches:
        raise LoopyError(
            "internal error: expected at least one CallKernel schedule item")

    # Generate WGSL source.
    from loopy.codegen import generate_code_v2
    if no_cache:
        # Avoid disk caching in restricted environments.
        from loopy import CacheMode
        with CacheMode(False):
            wgsl_source = generate_code_v2(t_unit).device_code()
    else:
        wgsl_source = generate_code_v2(t_unit).device_code()

    return WGSLWebGPUInfo(
        entrypoint=entrypoint,
        wgsl_source=wgsl_source,
        params=params_layout,
        storage_bindings=storage_bindings,
        dispatches=tuple(dispatches),
    )


def _pymbolic_expr_to_json(expr: Expression) -> object:
    """Convert a pymbolic expression to a JSON-serializable expression tree.

    This is intentionally small and only targets the kinds of expressions that
    appear in loopy-computed grid sizes (e.g. ceildiv patterns).
    """
    if isinstance(expr, bool):
        return bool(expr)
    if isinstance(expr, int):
        return int(expr)
    if isinstance(expr, float):
        return float(expr)

    if isinstance(expr, np.generic):
        if isinstance(expr, np.integer):
            return int(expr)
        if isinstance(expr, np.floating):
            return float(expr)

    if isinstance(expr, p.Variable):
        return {"var": expr.name}

    if isinstance(expr, p.Sum):
        return {"op": "add", "args": [_pymbolic_expr_to_json(child) for child in expr.children]}

    if isinstance(expr, p.Product):
        return {"op": "mul", "args": [_pymbolic_expr_to_json(child) for child in expr.children]}

    if isinstance(expr, p.FloorDiv):
        return {
            "op": "floordiv",
            "a": _pymbolic_expr_to_json(expr.numerator),
            "b": _pymbolic_expr_to_json(expr.denominator),
        }

    if isinstance(expr, p.Quotient):
        return {
            "op": "div",
            "a": _pymbolic_expr_to_json(expr.numerator),
            "b": _pymbolic_expr_to_json(expr.denominator),
        }

    if isinstance(expr, p.Remainder):
        return {
            "op": "mod",
            "a": _pymbolic_expr_to_json(expr.numerator),
            "b": _pymbolic_expr_to_json(expr.denominator),
        }

    if isinstance(expr, p.Min):
        return {"op": "min", "args": [_pymbolic_expr_to_json(child) for child in expr.children]}

    if isinstance(expr, p.Max):
        return {"op": "max", "args": [_pymbolic_expr_to_json(child) for child in expr.children]}

    raise LoopyError(
        "WGSLTarget WebGPU info cannot serialize expression of type "
        f"{type(expr).__name__}: {expr!r}")


def wgsl_webgpu_info_to_dict(info: WGSLWebGPUInfo) -> dict[str, object]:
    """Convert :class:`WGSLWebGPUInfo` to a JSON-serializable dict."""
    params: dict[str, object] | None
    if info.params is None:
        params = None
    else:
        params = {
            "group": info.params.group,
            "binding": info.params.binding,
            "struct_name": info.params.struct_name,
            "var_name": info.params.var_name,
            "size": info.params.size,
            "align": info.params.align,
            "fields": [
                {
                    "name": f.name,
                    "wgsl_type": f.wgsl_type,
                    "offset": f.offset,
                    "size": f.size,
                    "align": f.align,
                }
                for f in info.params.fields
            ],
        }

    storage_bindings = [
        {
            "group": b.group,
            "binding": b.binding,
            "name": b.name,
            "struct_name": b.struct_name,
            "access": b.access,
            "webgpu_buffer_binding_type": b.webgpu_buffer_binding_type,
            "element_type": b.element_type,
            "element_nbytes": b.element_nbytes,
        }
        for b in info.storage_bindings
    ]

    dispatches = [
        {
            "entrypoint": d.entrypoint,
            "workgroup_size": list(d.workgroup_size),
            "workgroups": [_pymbolic_expr_to_json(w) for w in d.workgroups],
        }
        for d in info.dispatches
    ]

    return {
        "entrypoint": info.entrypoint,
        "wgsl_source": info.wgsl_source,
        "params": params,
        "storage_bindings": storage_bindings,
        "dispatches": dispatches,
    }


# }}}


# vim: foldmethod=marker
