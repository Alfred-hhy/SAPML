
from Compiler.types import sint, cint, Array
from Compiler.library import print_ln, for_range, for_range_multithread, multithread, get_program, if_, if_then, else_then, end_if

NUM_BITS = 256


def reveal_all(secret_x, label):
    print_ln(f"{label}: %s", secret_x.reveal())


bits = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1]

def mod_inv(n):
    result = sint.Matrix(1, 1)
    result[0][0] = sint(1)


    store_base = sint.Matrix(1, 1)
    store_base[0][0] = n

    for i in range(len(bits)):
        cond = (bits[i] == 1)
        result[0][0] = result[0][0] * store_base[0][0] * cond + (1 - cond) * result[0][0]
        store_base[0][0] = store_base[0][0] * store_base[0][0]


    return result[0][0]




class AffineCurvePoint(object):

    def __init__(self, x, y, a, b):

        self.x = x
        self.y = y
        self.a = a
        self.b = b





class ProjectiveCurvePoint(object):

    def __init__(self, x, y, z, a, b):
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b



    def _create(self, x, y, z):
        return ProjectiveCurvePoint(x, y, z, self.a, self.b)

    def is_on_curve(self):
        return (self.y * self.y * self.z == self.x * self.x * self.x + self.a * self.x * self.z * self.z + self.b * self.z * self.z * self.z)



    def __add__(self, other):
        rx_val = sint.Matrix(1, 1)
        rx_val[0][0] = sint(0)

        ry_val = sint.Matrix(1, 1)
        ry_val[0][0] = sint(0)
        rz_val = sint.Matrix(1, 1)
        rz_val[0][0] = sint(0)


        check_self = ((self.x == 0) + (self.y == 0)) == 2
        check_other = ((other.x == 0) + (other.y == 0)) == 2
        @if_(check_self.reveal())
        def body():
            rx_val[0][0] = other.x
            ry_val[0][0] = other.y
            rz_val[0][0] = other.z

        @if_(check_other.reveal())
        def body():
            rx_val[0][0] = self.x
            ry_val[0][0] = self.y
            rz_val[0][0] = self.z


        none_check = ((self.x != sint(0)) + (other.x != sint(0)))
        @if_((none_check == 2).reveal())
        def body():
            t0 = self.y * other.z
            t1 = other.y * self.z
            u0 = self.x * other.z
            u1 = other.x * self.z
            @if_((u0 == u1).reveal())
            def body():
                if_then((t0 == t1).reveal())
                temp_double = self.double()
                rx_val[0][0] = temp_double.x
                ry_val[0][0] = temp_double.y
                rz_val[0][0] = temp_double.z
                else_then()
                rx_val[0][0] = sint(0)
                ry_val[0][0] = sint(0)
                rz_val[0][0] = sint(0)
                end_if()


            t = t0 - t1
            u = u0 - u1
            u2 = u * u
            v = self.z * other.z
            w = t * t * v - u2 * (u0 + u1)
            u3 = u * u2
            rx = u * w
            ry = t * (u0 * u2 - w) - t0 * u3
            rz = u3 * v

            cond = ((u0 == u1))
            @if_(rx_val[0][0] != None)
            def body():
                @if_((cond == 0).reveal())
                def body1():
                    rx_val[0][0] = rx
                    ry_val[0][0] = ry
                    rz_val[0][0] = rz

        return self._create(rx_val[0][0], ry_val[0][0], rz_val[0][0])

    def double(self):
        rx_val = sint.Matrix(1, 1)
        rx_val[0][0] = sint(1)
        ry_val = sint.Matrix(1, 1)
        ry_val[0][0] = sint(1)
        rz_val = sint.Matrix(1, 1)
        rz_val[0][0] = sint(1)


        none_check = ((self.x == 0) + (self.y == 0)) == 2
        @if_(none_check.reveal())
        def body():
            rx_val[0][0] = sint(0)
            ry_val[0][0] = sint(0)
            rz_val[0][0] = sint(0)

        @if_((self.y == 0).reveal())
        def body():
            rx_val[0][0] = sint(0)
            ry_val[0][0] = sint(0)
            rz_val[0][0] = sint(0)


        two = cint(2)
        t = self.x * self.x * cint(3) + self.a * self.z * self.z
        u = self.y * self.z * two
        v = u * self.x * self.y * two
        w = t * t - v * two
        rx = u * w
        ry = t * (v - w) - u * u * self.y * self.y * two
        rz = u * u * u


        @if_((rx_val[0][0] != sint(0)).reveal())
        def body():
            rx_val[0][0] = rx
            ry_val[0][0] = ry
            rz_val[0][0] = rz

        return self._create(rx_val[0][0], ry_val[0][0], rz_val[0][0])

    def __neg__(self):
        return self._create(self.x, -self.y, self.z)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, n):
        # Store x, y, z
        result = sint.Matrix(3, 1)
        result[0][0] = sint(0)
        result[1][0] = sint(0)
        result[2][0] = sint(0)

        temp = sint.Matrix(3, 1)
        temp[0][0] = self.x
        temp[1][0] = self.y
        temp[2][0] = self.z
        store_n = sint.Matrix(1, 1)
        store_n[0][0] = n



        @for_range(NUM_BITS)
        def loop_body(i):
            remainder = store_n[0][0] % 2
            @if_((remainder == sint(1)).reveal())
            def body():
                temp_point = ProjectiveCurvePoint(temp[0][0], temp[1][0], temp[2][0], a, b)
                temp_res = ProjectiveCurvePoint(result[0][0], result[1][0], result[2][0], a, b)

                temp_add = temp_res + temp_point
                result[0][0] = temp_add.x
                result[1][0] = temp_add.y
                result[2][0] = temp_add.z




            temp_point = ProjectiveCurvePoint(temp[0][0], temp[1][0], temp[2][0], a, b)
            temp_double = temp_point.double()
            temp[0][0] = temp_double.x
            temp[1][0] = temp_double.y
            temp[2][0] = temp_double.z
            store_n[0][0] = store_n[0][0] >> 1


        return ProjectiveCurvePoint(result[0][0], result[1][0], result[2][0], a, b)

    def to_affine_point(self):

        div = mod_inv(self.z)

        reveal_all(self.z, "self.z before to_affine")
        reveal_all(div, "z^{-1}")
        return AffineCurvePoint(self.x * div, self.y * div, self.a, self.b)


    def __eq__(self, other):
        cond = (self.x * other.z == other.x * self.z) + (self.y * other.z ==
                                                         other.y * self.z) + (self.a == other.a) + (self.b == other.b)
        return cond == 1

a = cint(0)
b = cint(13)


def compute_commitment(x, r):
    gen_x = cint(6)
    gen_y = cint(8804219299324514492806343261084073560212908759422749787101710634723026910978)
    generator = ProjectiveCurvePoint(gen_x, gen_y, cint(1), a, b)
    print_ln("gen_x: %s", generator.x)
    print_ln("gen_y %s", generator.y)
    inputs = sint.Matrix(1, 1)
    inputs[0][0] = 2

    q = sint(7)
    H = generator * q
    H = H * r

    a_times_G = generator * x


    commitment = a_times_G + H

    commitment = commitment.to_affine_point()
    print_ln("x_curve_point: %s", commitment.x.reveal())
    print_ln("y_curve_point: %s", commitment.y.reveal())

    return commitment
