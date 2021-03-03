import numpy as np

class vec:
	pass

class vec2(vec):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	# def __array__(self):
		# pass

	# def __array_wrap__(self):
		# pass

class vec3(vec):
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	# def __array__(self):
		# pass

	# def __array_wrap__(self):
		# pass


class char2(object):
	def __init__(self, x, y):
		x = np.int8(x)
		y = np.int8(y)

		super().__init__(x,y)

class short2(object):
	def __init__(self, x, y):
		x = np.int16(x)
		y = np.int16(y)

		super().__init__(x,y)

class int2(object):
	def __init__(self, x, y):
		x = np.int32(x)
		y = np.int32(y)

		super().__init__(x,y)

class long2(object):
	def __init__(self, x, y):
		x = np.int64(x)
		y = np.int64(y)

		super().__init__(x,y)

class uchar2(object):
	def __init__(self, x, y):
		x = np.uint8(x)
		y = np.uint8(y)

		super().__init__(x,y)

class ushort2(object):
	def __init__(self, x, y):
		x = np.unit16(x)
		y = np.unit16(y)

		super().__init__(x,y)

class uint2(object):
	def __init__(self, x, y):
		x = np.uint32(x)
		y = np.float32(y)

		super().__init__(x,y)

class ulong2(object):
	def __init__(self, x, y):
		x = np.uint64(x)
		y = np.uint64(y)

		super().__init__(x,y)

class float2(object):
	def __init__(self, x, y):
		x = np.float32(x)
		y = np.float32(y)

		super().__init__(x,y)

class double2(object):
	def __init__(self, x, y):
		x = np.float64(x)
		y = np.float64(y)

		super().__init__(x,y)




class char3(object):
	def __init__(self, x, y, z):
		x = np.int8(x)
		y = np.int8(y)
		z = np.int8(z)

		super().__init__(x,y,z)

class short3(object):
	def __init__(self, x, y, z):
		x = np.int16(x)
		y = np.int16(y)
		z = np.int16(z)

		super().__init__(x,y,z)

class int3(object):
	def __init__(self, x, y, z):
		x = np.int32(x)
		y = np.int32(y)
		z = np.int32(z)

		super().__init__(x,y,z)

class long3(object):
	def __init__(self, x, y, z):
		x = np.int64(x)
		y = np.int64(y)
		z = np.int64(z)

		super().__init__(x,y,z)

class uchar3(object):
	def __init__(self, x, y, z):
		x = np.uint8(x)
		y = np.uint8(y)
		z = np.uint8(z)

		super().__init__(x,y,z)

class ushort3(object):
	def __init__(self, x, y, z):
		x = np.unit16(x)
		y = np.unit16(y)
		z = np.unit16(z)

		super().__init__(x,y,z)

class uint3(object):
	def __init__(self, x, y, z):
		x = np.uint32(x)
		y = np.float32(y)
		z = np.float32(z)

		super().__init__(x,y,z)

class ulong3(object):
	def __init__(self, x, y, z):
		x = np.uint64(x)
		y = np.uint64(y)
		z = np.uint64(z)

		super().__init__(x,y,z)

class float3(object):
	def __init__(self, x, y, z):
		x = np.float32(x)
		y = np.float32(y)
		z = np.float32(z)

		super().__init__(x,y,z)

class double3(object):
	def __init__(self, x, y, z):
		x = np.float64(x)
		y = np.float64(y)
		z = np.float64(z)

		super().__init__(x,y,z)
