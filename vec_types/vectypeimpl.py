from numba import types
from numba.core.extending import typeof_impl, type_callable
from numba.core.typing import signature

import operator
# from llvmlite.llvmpy.core import Type

from numba.core.extending import models, register_model, lower_builtin #, make_attribute_wrapper
from numba.core import cgutils
from numba.cpython import mathimpl

from numba.cuda.cudadecl import registry as decl_registry
from numba.cuda.cudaimpl import registry as impl_registry
from numba.core.typing.templates import ConcreteTemplate, AttributeTemplate

from vectype import ( char2,  short2,  int2,  long2,
					 uchar2, ushort2, uint2, ulong2,
					 float2, double2,
					 char3,  short3,  int3,  long3,
					 uchar3, ushort3, uint3, ulong3,
					 float3, double3)
from vectypedecl import Vec2, Vec3
from vectypedecl import (char2 as char2_type,
						 short2 as short2_type,
						 int2 as int2_type,
						 long2 as long2_type,
						 uchar2 as uchar2_type,
						 ushort2 as ushort2_type,
						 uint2 as uint2_type,
						 ulong2 as ulong2_type,
						 float2 as float2_type,
						 double2 as double2_type,
						 char3 as char3_type,
						 short3 as short3_type,
						 int3 as int3_type,
						 long3 as long3_type,
						 uchar3 as uchar3_type,
						 ushort3 as ushort3_type,
						 uint3 as uint3_type,
						 ulong3 as ulong3_type,
						 float3 as float3_type,
						 double3 as double3_type)

import cudafuncs

#python classes
vec2s = [uchar2, ushort2, uint2, ulong2,
		char2, short2, int2, long2,
		float2, double2]
vec3s = [uchar3, ushort3, uint3, ulong3,
		char3, short3, int3, long3,
		float3, double3]

#initialized numba type classes
vec2types = [uchar2_type, ushort2_type, uint2_type, ulong2_type,
			 char2_type, short2_type, int2_type, long2_type,
			 float2_type, double2_type]

vec3types = [uchar3_type, ushort3_type, uint3_type, ulong3_type,
			 char3_type, short3_type, int3_type, long3_type,
			 float3_type, double3_type]

vec_groups = [vec2s, vec3s]
vectype_groups = [vec2types, vec3types]
Vecs = [Vec2, Vec3]

#generic libfunc helper function for unary and binary vector ops
def libfunc_helper_wrapper(cudafuncs_op):
	def libfunc_helper(context, builder, sig, args):
		types = [sig.return_type.member_type]*(len(sig.args) + 1)
		sig = signature(*types)
		return context.get_function(cudafuncs_op, sig), libfunc_op_caller
	return libfunc_helper

#####################
#op caller functions#
#####################


#these are returned from a helper function along with the op(s) that the caller function will call
#builder, mathimpl, and libfunc each have their own calling convention so a unique caller is required for each

def builder_op_caller(op,builder,rtrn,*args):
	for n, attr in enumerate(rtrn._fe_type.member_names):
		setattr(rtrn, attr, op(*[arg[n] for arg in args]))
	return rtrn._getvalue()

def mathimpl_op_caller(op,builder,rtrn,*args):
	for n, attr in enumerate(rtrn._fe_type.member_names):
		setattr(rtrn, attr, op(builder, *[arg[n] for arg in args]))
	return rtrn._getvalue()

def libfunc_op_caller(op,builder,rtrn,*args):
	for n, attr in enumerate(rtrn._fe_type.member_names):
		setattr(rtrn, attr, op(builder, [arg[n] for arg in args]))
	return rtrn._getvalue()

###########
#UNARY OPS#
###########

def unary_op_factory(op, vectype, op_helper):
	@impl_registry.lower(op, vectype)
	def cuda_op_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.return_type)(context, builder, value = args[0])
		out = cgutils.create_struct_proxy(sig.return_type)(context, builder)

		a_members = [getattr(a, attr) for attr in sig.return_type.member_names]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,out,a_members)

########################
#unary helper functions#
########################

#a helper function defines the operation(s) to be called on the vector as well as returns the caller that will execute the operation(s)

def neg_helper(context, builder, sig, args):
	neg = builder.neg
	caller = builder_op_caller
	if isinstance(sig.return_type.member_type, types.Float):
		neg = mathimpl.negate_real
		caller = mathimpl_op_caller
	return neg, caller

	# #for when fneg is supported
	# neg = builder.neg
	# if isinstance(sig.return_type.member_type, types.Float):
	# 	neg = builder.fneg
	# return neg, builder_op_caller

############
#BINARY OPS#
############

def binary_op_factory_vec_vec(op, vectype, op_helper):
	@impl_registry.lower(op, vectype, vectype)
	def cuda_op_vec_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.return_type)(context, builder, value = args[0])
		b = cgutils.create_struct_proxy(sig.return_type)(context, builder, value = args[1])
		out = cgutils.create_struct_proxy(vectype)(context, builder)

		a_members = [getattr(a, attr) for attr in sig.return_type.member_names]
		b_members = [getattr(b, attr) for attr in sig.return_type.member_names]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,out,a_members,b_members)

def binary_op_factory_vec_scalar(op, vectype, op_helper):
	@impl_registry.lower(op, types.Number, vectype)
	@impl_registry.lower(op, vectype, types.Number)
	def cuda_op_vec_scalar(context, builder, sig, args):
		a = sig.args[1] == sig.return_type
		b = 1 - a

		a = cgutils.create_struct_proxy(vectype)(context, builder, value=args[a])
		b = context.cast(builder, args[b], sig.args[b], sig.return_type.member_type)
		out = cgutils.create_struct_proxy(vectype)(context, builder)

		a_members = [getattr(a, attr) for attr in sig.return_type.member_names]
		b_members = [b for attr in sig.return_type.member_names]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,out,a_members,b_members)

def binary_op_factory(op, vectype, op_helper):
	binary_op_factory_vec_vec(op, vectype, op_helper)
	binary_op_factory_vec_scalar(op, vectype, op_helper)

#########################
#binary helper functions#
#########################

def add_helper(context, builder, sig, args):
	add = builder.add
	if isinstance(sig.return_type.member_type, types.Float):
		add = builder.fadd
	return add, builder_op_caller

def sub_helper(context, builder, sig, args):
	sub = builder.sub
	if isinstance(sig.return_type.member_type, types.Float):
		sub = builder.fsub
	return sub, builder_op_caller

def mul_helper(context, builder, sig, args):
	mul = builder.mul
	if isinstance(sig.return_type.member_type, types.Float):
		mul = builder.fmul
	return mul, builder_op_caller

def div_helper(context, builder, sig, args):
	div = builder.udiv
	if isinstance(sig.return_type.member_type, types.Float):
		div = builder.fdiv
	elif sig.return_type.member_type.member_type.signed:
		div = builder.sdiv
	return div, builder_op_caller

# def mod_helper(context, builder, sig, args):
	# libfunc_impl = context.get_function(cudafuncs_op, sig)

	# # mod = 

	# # 	for n, attr in enumerate(vectype.member_names):
	# # 		setattr(out, attr, libfunc_impl(builder, [getattr(a, attr)]))
	# # 	# out.x = libfunc_impl(builder, [a.x])
	# # 	# out.y = libfunc_impl(builder, [a.y])

# def pow_helper(context, builder, sig, args):
	# pass
	#DONT WANT TO CAST SCALAR VALUE USED FOR POWER
	# out = cgutils.create_struct_proxy(vectype)(context, builder)

	# libfunc_impl = context.get_function(operator.pow, signature(ax, ax, ax))
	# out.x = libfunc_impl(builder, [ax, bx]) 
	# out.y = libfunc_impl(builder, [ay, by]) 

	# return out._getvalue()

###############
#REDUCTION OPS# (ops that take in a vector and return a scalar)
###############

def reduction_op_factory(op, vectype, op_helper):
	#functions that reduce a vector input to a scalar output by applying op to each element of the input vector
	@impl_registry.lower(op, vectype)
	def cuda_op_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0])(context, builder, value = args[0])
		a_members = [getattr(a, attr) for attr in sig.args[0].member_names]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,a_members)

#############################
#reduction callers functions#
#############################

def reduction_builder_op_caller(op,builder,vec):
	val = vec[0]
	for n in range(len(vec) - 1):
		val = op(val, vec[n + 1])
	return val

def reduction_libfunc_op_caller(op,builder,vec):
	val = vec[0]
	for n in range(len(vec) - 1):
		val = op(builder, [val, vec[n + 1]])
	return val

def length_caller(ops, builder, vec):
	add, mul, sqrt = ops
	val = mul(builder, [vec[0], vec[0]])
	for n in range(len(vec) - 1):
		val = add(builder, [val, mul(builder, [vec[n + 1], vec[n + 1]])])
	return sqrt(builder, [val])

#######################################
#reduction op helper functions#
#######################################

def sum_helper(context, builder, sig, args):
	add = builder.add
	if isinstance(sig.args[0].member_type, types.Float):
		add = builder.fadd
	return add, reduction_builder_op_caller

dot_helper = mul_helper

def length_helper(context, builder, sig, args):
	libsig = signature(sig.args[0].member_type,
					   sig.args[0].member_type)
	sqrt = context.get_function(cudafuncs.sqrt, libsig)

	libsig = signature(sig.args[0].member_type,
					   sig.args[0].member_type,
					   sig.args[0].member_type)
	mul = context.get_function(operator.mul, libsig)
	add = context.get_function(operator.add, libsig)

	return [add, mul, sqrt], length_caller

def reduction_libfunc_helper_wrapper(cudafuncs_op):
	def reduction_libfunc_helper(context, builder, sig, args):
		sig = signature(sig.args[0].member_type,
						sig.args[0].member_type,
						sig.args[0].member_type)
		return context.get_function(cudafuncs_op, sig), reduction_libfunc_op_caller
	return reduction_libfunc_helper

##################
#VECTOR FUNCTIONS#
##################

##############################
#define vector function stubs#
##############################

from cudadecl import register_op, unary_op_fixed_type_factory, binary_op_fixed_type_factory

def sum(vec):
	pass

def dot(vec):
	pass

def length(vec):
	pass

# def cross(veca, vecb):
# 	pass

# def lerp(t, veca, vecb):
# 	# return a + t*(b-a);
# 	pass

# interp = lerp #np.interp

# def clamp(vec, lo, hi):
# 	pass

# clip = clamp #np.clip

# def normalize(vec):
# 	# float invLen = rsqrtf(dot(v, v));
# 	# return v * invLen;
# 	pass

# def frac(vec):
# 	pass

# def mod(veca, vecb):
#     # return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
# 	pass

#insert functions into cudafuncs if they don't already exist (because functions aren't typed only a single instance of a function with the given name needs to exist within cudafuncs)
vec_funcs = [(sum, unary_op_fixed_type_factory, None),
			 (dot, binary_op_fixed_type_factory, None),
			 (length, unary_op_fixed_type_factory, None)]

for vec_func in vec_funcs:
	func, factory, *factory_args = vec_func
	name = func.__name__
	libfunc = getattr(cudafuncs, name, None)
	#only register a vector function if a stub doesn't already exist within cudafuncs
	if libfunc is None:
		setattr(cudafuncs, name, func)
		#note: None is set as default return type for some of the vec_funcs so that a typer_ method must be defined to use the op on input types
		register_op(name, factory, *factory_args)


def vecs_factory(vecs, vectypes, Vec):
	def vec_factory(vec, vectype, Vec):
		VecType = type(vectype)

		##########################################
		#Define Vector Type
		##########################################

		#note: the typer function must have the same number of arguments as inputs given to the struct so each initialize style needs its own typer

		@typeof_impl.register(vec)
		def typeof_vec(val, c):
			return vectype

		def vec_typer(*attrs):
			if all([isinstance(attr, types.Number) for attr in attrs]):
				return vectype
			else:
				raise ValueError(f"Input to {vec.__name__} not understood")

		#vector length specific initialization
		if Vec == Vec2:
			#initialize: vec2(x,y)
			@type_callable(vec)
			def type_vec2(context):
				def typer(x, y):
					return vec_typer(x,y)
				return typer
		elif Vec == Vec3:
			#initialize: vec3(x,y,z)
			@type_callable(vec)
			def type_vec3(context):
				def typer(x, y, z):
					return vec_typer(x,y,z)
				return typer

			#initialize: vec3(vec2,y) or vec3(x,vec2)
			@type_callable(vec)
			def type_vec3(context):
				def typer(vec_or_scalar_a, vec_or_scalar_b):
					if isinstance(vec_or_scalar_a, types.Number) and isinstance(vec_or_scalar_b, Vec2):
						return vectype
					elif isinstance(vec_or_scalar_a, Vec2) and isinstance(vec_or_scalar_b, types.Number):
						return vectype
					else:
						raise ValueError(f"Input to {vec.__name__} not understood")
				return typer

		#initialize: vec(x) or vec(vec)
		@type_callable(vec)
		def type_vec(context):
			def typer(vec_or_scalar):
				if isinstance(vec_or_scalar, types.Number) or isinstance(vec_or_scalar, Vec):
					return vectype
				else:
					raise ValueError(f"Input to {vec.__name__} not understood")
			return typer

		#initialize: vec()
		@type_callable(vec)
		def type_vec(context):
			def typer():
				return vectype
			return typer

		@register_model(VecType) 
		class VecModel(models.StructModel):
			def __init__(self, dmm, fe_type):
				models.StructModel.__init__(self, dmm, fe_type, vectype.members)

		##########################################
		#Initializer/Constructor Methods
		##########################################

		#initialize: vec(vec)
		@lower_builtin(vec, Vec)
		def impl_vec(context, builder, sig, args):
			a = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])
			out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
			for n, attr in enumerate(vectype.member_names):
				setattr(out, attr, context.cast(builder, getattr(a, attr), sig.args[0].member_type, sig.return_type.member_type))
			return out._getvalue()

		#initialize: vec(x,y,...)
		@lower_builtin(vec, *[types.Number]*len(vectype.members))
		def impl_vec(context, builder, sig, args):
			out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
			for n, attr in enumerate(vectype.member_names):
				setattr(out, attr, context.cast(builder, args[n], sig.args[n], sig.return_type.member_type))
			return out._getvalue()

		#initialize: vec(x)
		@lower_builtin(vec, types.Number)
		def impl_vec(context, builder, sig, args):
			out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
			cast = context.cast(builder, args[0], sig.args[0], sig.return_type.member_type)
			for n, attr in enumerate(vectype.member_names):
				setattr(out, attr, cast)
			return out._getvalue()

		#initialize: vec()
		@lower_builtin(vec)
		def impl_vec(context, builder, sig, args):
			out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
			const = context.get_constant(sig.return_type.member_type, 0)
			for n, attr in enumerate(vectype.member_names):
				setattr(out, attr, const)
			return out._getvalue()

		if Vec == Vec3:
			#initialize: vec(x, vec2)
			@lower_builtin(vec, Vec2, types.Number)
			def impl_vec(context, builder, sig, args):
				vec = cgutils.create_struct_proxy(sig.args[0])(context, builder, value = args[0])
				out = cgutils.create_struct_proxy(sig.return_type)(context, builder)

				out.x = context.cast(builder, vec.x, sig.args[0].member_type, sig.return_type.member_type)
				out.y = context.cast(builder, vec.y, sig.args[0].member_type, sig.return_type.member_type)
				out.z = context.cast(builder, args[1], sig.args[1], sig.return_type.member_type)

				return out._getvalue()

			#initialize: vec(vec2, x)
			@lower_builtin(vec, types.Number, Vec2)
			def impl_vec(context, builder, sig, args):
				vec = cgutils.create_struct_proxy(sig.args[1])(context, builder, value = args[1])
				out = cgutils.create_struct_proxy(sig.return_type)(context, builder)

				out.x = context.cast(builder, args[0], sig.args[0], sig.return_type.member_type)
				out.y = context.cast(builder, vec.x, sig.args[1].member_type, sig.return_type.member_type)
				out.z = context.cast(builder, vec.y, sig.args[1].member_type, sig.return_type.member_type)

				return out._getvalue()

		##########################################
		#Define Vector Attributes
		##########################################

		@decl_registry.register_attr
		class Vec_attrs(AttributeTemplate):
			key = vectype

			def resolve_x(self, mod):
				return vectype.member_type

			def resolve_y(self, mod):
				return vectype.member_type

			def resolve_z(self, mod):
				return vectype.member_type

		@impl_registry.lower_getattr(vectype, 'x')
		def vec_get_x(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
			return vec.x

		# #SETATTR CURRENTLY NOT SUPPORTED ON GPU
		# @impl_registry.lower_setattr(vectype, 'x')
		# def vec_set_x(context, builder, sig, args):
			# typ, valty = sig.args
			# target, val = args

			# vec = cgutils.create_struct_proxy(typ)(context, builder, value=target)
			# val = context.cast(builder, val, valty, typ.member_type)
			# out = cgutils.create_struct_proxy(typ)(context, builder)

			# return setattr(out, 'x', val)

		@impl_registry.lower_getattr(vectype, 'y')
		def vec_get_y(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
			return vec.y

		# @impl_registry.lower_setattr(vectype, 'y')
		# def vec_set_y(context, builder, sig, args):
		# 	pass

		@impl_registry.lower_getattr(vectype, 'z')
		def vec_get_z(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
			return vec.z

		# @impl_registry.lower_setattr(vectype, 'z')
		# def vec_set_z(context, builder, sig, args):
		# 	pass

		#.xy, .xz, .yz methods
		if Vec == Vec3:
			@decl_registry.register_attr
			class Vec3_attrs(AttributeTemplate):
				key = vectype

				def resolve_xy(self, mod):
					return vectype.vec2

				def resolve_xz(self, mod):
					return vectype.vec2

				def resolve_yz(self, mod):
					return vectype.vec2

			@impl_registry.lower_getattr(vectype, 'xy')
			def vec3_get_xy(context, builder, sig, args):
				vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
				xy = cgutils.create_struct_proxy(sig.vec2)(context, builder)
				xy.x = vec.x
				xy.y = vec.y
				return xy._getvalue()

			# @impl_registry.lower_setattr(vectype, 'xy')
			# def vec3_set_xy(context, builder, sig, args):
				# pass
				# #scalar applies to every input
				# #vec2 applies to each component

			@impl_registry.lower_getattr(vectype, 'xz')
			def vec3_get_xz(context, builder, sig, args):
				vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
				xz = cgutils.create_struct_proxy(sig.vec2)(context, builder)
				xz.x = vec.x
				xz.y = vec.z
				return xz._getvalue()

			# @impl_registry.lower_setattr(vectype, 'xz')
			# def vec3_set_xz(context, builder, sig, args):
			# 	pass

			@impl_registry.lower_getattr(vectype, 'yz')
			def vec3_get_yz(context, builder, sig, args):
				vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
				yz = cgutils.create_struct_proxy(sig.vec2)(context, builder)
				yz.x = vec.y
				yz.y = vec.z
				return yz._getvalue()

			# @impl_registry.lower_setattr(vectype, 'yz')
			# def vec3_set_yz(context, builder, sig, args):
			# 	pass

		##########################################
		#Register Vector Methods
		##########################################

		#####
		#+, -, *, /
		#####

		class vec_op_template(ConcreteTemplate):
			cases = [signature(vectype, vectype, vectype),
					 signature(vectype, vectype.member_type, vectype),
					 signature(vectype, vectype, vectype.member_type)]

		ops = [(operator.add, add_helper),
			   (operator.sub, sub_helper),
			   (operator.mul, mul_helper),
			   (operator.truediv, div_helper)]

		for op, helper in ops:
			decl_registry.register_global(op)(vec_op_template)
			binary_op_factory(op, vectype, helper)

		#####
		#+=, -=, *=, /= (iadd,isub,imul,itruediv)
		#####

		#TO DO

		#####
		#% (should this be vectype specific?)
		#####

		#TO DO
		#"fmod"

		#####
		#** (should this be vectype specific?)
		#####

		#TO DO
		#"pow", "powi"

		#####
		#**=, %= (ipow,imod)
		#####

		#TO DO

		#####
		#min, max, sum
		#####

		class vec_op_template(ConcreteTemplate):
					 #binary
			cases = [signature(vectype, vectype, vectype),
					 signature(vectype, vectype.member_type, vectype),
					 signature(vectype, vectype, vectype.member_type),
					 #reduction
					 signature(vectype.member_type, vectype)]

		decl_registry.register_global(min)(vec_op_template)
		decl_registry.register_global(max)(vec_op_template)
		# decl_registry.register_global(sum)(vec_op_template)

		min_helper = libfunc_helper_wrapper(cudafuncs.min)
		max_helper = libfunc_helper_wrapper(cudafuncs.max)

		binary_op_factory(min, vectype, min_helper)
		binary_op_factory(max, vectype, max_helper)
		binary_op_factory(cudafuncs.min, vectype, min_helper)
		binary_op_factory(cudafuncs.max, vectype, max_helper)

		min_helper = reduction_libfunc_helper_wrapper(cudafuncs.min)
		max_helper = reduction_libfunc_helper_wrapper(cudafuncs.max)

		reduction_op_factory(min, vectype, min_helper)
		reduction_op_factory(max, vectype, max_helper)
		reduction_op_factory(cudafuncs.min, vectype, min_helper)
		reduction_op_factory(cudafuncs.max, vectype, max_helper)

		# reduction_op_factory(sum, vectype, sum_helper)
		reduction_op_factory(cudafuncs.sum, vectype, sum_helper)
		reduction_op_factory(cudafuncs.length, vectype, length_helper)
		binary_op_factory_vec_vec(cudafuncs.dot, vectype, dot_helper)

	for vec, vectype in zip(vecs, vectypes):
		vec_factory(vec, vectype, Vec)

	##########################################
	#Vector Type Specific Methods
	##########################################

	#####
	#umin, ullmin, llmin, fminf, fmin, umax, ullmax, llmax, fmaxf, fmax
	#####

	#uint/ulong/long/float/double vectors
	for op, vectype in zip([cudafuncs.umin, cudafuncs.ullmin, cudafuncs.llmin, cudafuncs.fminf, cudafuncs.fmin],
						   (vectypes[2:4] + vectypes[6:10])):
		binary_op_factory(op, vectype, libfunc_helper_wrapper(op))
	for op, vectype in zip([cudafuncs.umax, cudafuncs.ullmax ,cudafuncs.llmax, cudafuncs.fmaxf, cudafuncs.fmax],
						   (vectypes[2:4] + vectypes[6:10])):
		binary_op_factory(op, vectype, libfunc_helper_wrapper(op))

	#only define negate/abs for vectors that can be negative
	for vectype in vectypes[4:]:
		#####
		#- (neg), abs
		#####

		class vec_unary_op(ConcreteTemplate):
			cases = [signature(vectype, vectype)]

		decl_registry.register_global(operator.neg)(vec_unary_op)
		unary_op_factory(operator.neg, vectype, neg_helper)

		abs_helper = libfunc_helper_wrapper(cudafuncs.abs)

		decl_registry.register_global(abs)(vec_unary_op)
		unary_op_factory(abs, vectype, abs_helper)
		unary_op_factory(cudafuncs.abs, vectype, abs_helper)

	#####
	#llabs, fabs, fabsf
	#####

	#long/float/double vector
	for op, vectype in zip([cudafuncs.llabs, cudafuncs.fabs, cudafuncs.fabsf],
							vectypes[7:]):
		unary_op_factory(op, vectype, libfunc_helper_wrapper(op))

	#####
	#unary functions
	#####

	ops = ["sin", "cos", "tan",
		   "sinh", "cosh", "tanh",
		   "asin", "acos", "atan",
		   "asinh", "acosh", "atanh",
		   "sqrt", "rsqrt", "cbrt", "rcbrt",
		   "exp" ,"exp10", "exp2", "expm1",
		   "log", "log1p", "log10", "log2", "logb",
		   "floor"]

	#float/double vector
	for vectype in vectypes[8:]:
		for op in ops:
			libop = getattr(cudafuncs, op)
			unary_op_factory(libop, vectype, libfunc_helper_wrapper(libop))

		#####
		#round
		#####

		class vec_unary_op(ConcreteTemplate):
			cases = [signature(vectype, vectype)]

		round_helper = libfunc_helper_wrapper(cudafuncs.round)

		decl_registry.register_global(round)(vec_unary_op)
		unary_op_factory(round, vectype, round_helper)
		unary_op_factory(cudafuncs.round, vectype, round_helper)

	#float vector
	for vectype in vectypes[8:9]:
		for op in ops:
			libopf = getattr(cudafuncs, op + 'f')
			unary_op_factory(libopf, vectype, libfunc_helper_wrapper(libopf))


	#####
	#binary functions
	#####

	# atan2, cross, lerp, etc.




for vecs, vectypes, Vec in zip(vec_groups, vectype_groups, Vecs):
	vecs_factory(vecs, vectypes, Vec)
