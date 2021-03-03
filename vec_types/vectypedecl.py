from numba import types

#To add typer behaviour for a given function and vector type add typer_FUNCTION and define the return type for the function

class Vec(types.Type):
	def __init__(self, name = 'vec'):
		super().__init__(name = name)

	@property
	def members(self):
		return [(name, self.member_type)for name in self.member_names]

	def typer_sum(self, context):
		return self.member_type

	def typer_dot(self, context, other):
		return self

	def typer_min(self, context, *other):
		#if unary max is called a scalar is returned
		if not other:
			return self.member_type
		return self

	def typer_max(self, context, *other):
		#if unary min is called a scalar is returned
		if not other:
			return self.member_type
		return self

class IntegerVec(Vec):
	def __init__(self, name = 'integer_vec'):
		super().__init__(name = name)

class CharVec(IntegerVec):
	def __init__(self, name = 'char_vec'):
		super().__init__(name = name)
		self.member_type = types.int8

class ShortVec(IntegerVec):
	def __init__(self, name = 'short_vec'):
		super().__init__(name = name)
		self.member_type = types.int16

class IntVec(IntegerVec):
	def __init__(self, name = 'int_vec'):
		super().__init__(name = name)
		self.member_type = types.int32

	def typer_abs(self, context, other):
		return self

class LongVec(IntegerVec):
	def __init__(self, name = 'long_vec'):
		super().__init__(name = name)
		self.member_type = types.int64

	def typer_llabs(self, context):
		return self

	def typer_llmin(self, context, *other):
		if not other:
			return self.member_type
		return self
	def typer_llmax(self, context, *other):
		if not other:
			return self.member_type
		return self

class UCharVec(IntegerVec):
	def __init__(self, name = 'uchar_vec'):
		super().__init__(name = name)
		self.member_type = types.uint8

class UShortVec(IntegerVec):
	def __init__(self, name = 'ushort_vec'):
		super().__init__(name = name)
		self.member_type = types.uint16

class UIntVec(IntegerVec):
	def __init__(self, name = 'uint_vec'):
		super().__init__(name = name)
		self.member_type = types.uint32

	def typer_umin(self, context, *other):
		if not other:
			return self.member_type
		return self
	def typer_umax(self, context, *other):
		if not other:
			return self.member_type
		return self

class ULongVec(IntegerVec):
	def __init__(self, name = 'ulong_vec'):
		super().__init__(name = name)
		self.member_type = types.uint64

	def typer_ullmin(self, context, *other):
		if not other:
			return self.member_type
		return self
	def typer_ullmax(self, context, *other):
		if not other:
			return self.member_type
		return self

class FloatingVec(Vec):
	def __init__(self, name = 'floating_vec'):
		super().__init__(name = name)

	def typer_round(self, context):
		return self

	def typer_abs(self, context, other):
		return self
	def typer_sin(self, context):
		return self
	def typer_cos(self, context):
		return self
	def typer_tan(self, context):
		return self
	def typer_asin(self, context):
		return self
	def typer_acos(self, context):
		return self
	def typer_atan(self, context):
		return self
	def typer_sinh(self, context):
		return self
	def typer_cosh(self, context):
		return self
	def typer_tanh(self, context):
		return self
	def typer_asinh(self, context):
		return self
	def typer_acosh(self, context):
		return self
	def typer_atanh(self, context):
		return self
	def typer_sinpi(self, context):
		return self
	def typer_cospi(self, context):
		return self
	# # def typer_atan2(self, context):
	# #	return 
	# # def typer_sincos(self, context):
	# #	return 
	# # def typer_sincospi(self, context):
	# #	return 

	def typer_sqrt(self, context):
		return self
	def typer_rsqrt(self, context):
		return self
	def typer_cbrt(self, context):
		return self
	def typer_rcbrt(self, context):
		return self
	def typer_exp(self, context):
		return self
	def typer_exp10(self, context):
		return self
	def typer_exp2(self, context):
		return self
	def typer_expm1(self, context):
		return self
	def typer_log(self, context):
		return self
	def typer_log1p(self, context):
		return self
	def typer_log10(self, context):
		return self
	def typer_log2(self, context):
		return self
	def typer_logb(self, context):
		return self

class FloatVec(FloatingVec):
	def __init__(self, name = 'float_vec'):
		super().__init__(name = name)
		self.member_type = types.float32

	def typer_length(self, context):
		return self.member_type

	def typer_fabsf(self, context):
		return self

	def typer_fminf(self, context, *other):
		if not other:
			return self.member_type
		return self
	def typer_fmaxf(self, context, *other):
		if not other:
			return self.member_type
		return self

	def typer_sinf(self, context):
		return self
	def typer_cosf(self, context):
		return self
	def typer_tanf(self, context):
		return self
	def typer_asinf(self, context):
		return self
	def typer_acosf(self, context):
		return self
	def typer_atanf(self, context):
		return self
	def typer_sinhf(self, context):
		return self
	def typer_coshf(self, context):
		return self
	def typer_tanhf(self, context):
		return self
	def typer_asinhf(self, context):
		return self
	def typer_acoshf(self, context):
		return self
	def typer_atanhf(self, context):
		return self
	# def typer_sinpif(self, context):
	#	return 
	# def typer_cospif(self, context):
	#	return 
	# def typer_atan2f(self, context):
	#	return 
	# def typer_sincosf(self, context):
	#	return 
	# def typer_sincospif(self, context):
	#	return 
	def typer_sqrtf(self, context):
		return self
	def typer_rsqrtf(self, context):
		return self
	def typer_cbrtf(self, context):
		return self
	def typer_rcbrtf(self, context):
		return self
	def typer_expf(self, context):
		return self
	def typer_exp10f(self, context):
		return self
	def typer_exp2f(self, context):
		return self
	def typer_expm1f(self, context):
		return self
	def typer_logf(self, context):
		return self
	def typer_log1pf(self, context):
		return self
	def typer_log10f(self, context):
		return self
	def typer_log2f(self, context):
		return self
	def typer_logbf(self, context):
		return self

	def typer_fast_sinf(self, context):
		return self
	def typer_fast_cosf(self, context):
		return self
	def typer_fast_tanf(self, context):
		return self
	def typer_fast_expf(self, context):
		return self
	def typer_fast_exp10f(self, context):
		return self
	def typer_fast_logf(self, context):
		return self
	def typer_fast_log10f(self, context):
		return self
	def typer_fast_log2f(self, context):
		return self

	# def typer_fast_sincosf(self, context):
	# 	return self


class DoubleVec(FloatingVec):
	def __init__(self, name = 'double_vec'):
		super().__init__(name = name)
		self.member_type = types.float64

	def typer_fabs(self, context):
		return self

	def typer_fmin(self, context, *other):
		if not other:
			return self.member_type
		return self
	def typer_fmax(self, context, *other):
		if not other:
			return self.member_type
		return self




class Vec2(Vec):
	def __init__(self,  name = 'vec2'):
		super().__init__(name = name)
		self.member_names = ['x', 'y']

class Vec3(Vec):
	def __init__(self, name = 'vec3'):
		super().__init__(name = name)
		self.member_names = ['x', 'y', 'z']




class UChar2(Vec2, UCharVec):
	def __init__(self):
		super().__init__(name = 'uchar2')

class UShort2(Vec2, UShortVec):
	def __init__(self):
		super().__init__(name = 'ushort2')

class UInt2(Vec2, UIntVec):
	def __init__(self):
		super().__init__(name = 'uint2')

class ULong2(Vec2, ULongVec):
	def __init__(self):
		super().__init__(name = 'ulong2')

class Char2(Vec2, CharVec):
	def __init__(self):
		super().__init__(name = 'char2')

class Short2(Vec2, ShortVec):
	def __init__(self):
		super().__init__(name = 'short2')

class Int2(Vec2, IntVec):
	def __init__(self):
		super().__init__(name = 'int2')

class Long2(Vec2, LongVec):
	def __init__(self):
		super().__init__(name = 'long2')

class Float2(Vec2, FloatVec):
	def __init__(self):
		super().__init__(name = 'float2')

class Double2(Vec2, DoubleVec):
	def __init__(self):
		super().__init__(name = 'double2')



class UChar3(Vec3, UCharVec):
	def __init__(self):
		super().__init__(name = 'uchar3')
		self.vec2 = char2

class UShort3(Vec3, UShortVec):
	def __init__(self):
		super().__init__(name = 'ushort3')
		self.vec2 = short2

class UInt3(Vec3, UIntVec):
	def __init__(self):
		super().__init__(name = 'uint3')
		self.vec2 = int2

class ULong3(Vec3, ULongVec):
	def __init__(self):
		super().__init__(name = 'ulong3')
		self.vec2 = long2

class Char3(Vec3, CharVec):
	def __init__(self):
		super().__init__(name = 'char3')
		self.vec2 = uchar2

class Short3(Vec3, ShortVec):
	def __init__(self):
		super().__init__(name = 'short3')
		self.vec2 = ushort2

class Int3(Vec3, IntVec):
	def __init__(self):
		super().__init__(name = 'int3')
		self.vec2 = uint2

class Long3(Vec3, LongVec):
	def __init__(self):
		super().__init__(name = 'long3')
		self.vec2 = ulong2

class Float3(Vec3, FloatVec):
	def __init__(self):
		super().__init__(name = 'float3')
		self.vec2 = float2

class Double3(Vec3, DoubleVec):
	def __init__(self):
		super().__init__(name = 'double3')
		self.vec2 = double2


vec = Vec()

vec2 = Vec2()
vec3 = Vec3()

char2 = Char2()
short2 = Short2()
int2 = Int2()
long2 = Long2()
uchar2 = UChar2()
ushort2 = UShort2()
uint2 = UInt2()
ulong2 = ULong2()
float2 = Float2()
double2 = Double2()

char3 = Char3()
short3 = Short3()
int3 = Int3()
long3 = Long3()
uchar3 = UChar3()
ushort3 = UShort3()
uint3 = UInt3()
ulong3 = ULong3()
float3 = Float3()
double3 = Double3()
