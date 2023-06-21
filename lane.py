import matplotlib.pyplot as plt

kMaxNumberOfDiscretePoints = 100
kDistanceBetweenLinePoints = 3

class Polynomial:
    def __init__(self, order):
        self.order = order + 1
        self.coeff_ls = [0]*self.order

    def set_coeffs(self, i, val):
        try:
            self.coeff_ls[i] = val
        except IndexError:
            print("Out of range for Polynomial!")

    def evaluate(self, x_point):
        """
        @brief Evaluates the polynomial in a given point.
        @param x_point: The point in which the polynomial is evaluated
        @return: Polynomial(x_point) representing the value of the polynomial in the given point.
        """
        y_value = 0.0
        x = 1.0
        for coeff in self.coeff_ls:
            y_value += x*coeff
            x *= x_point
        return y_value

class LaneGenerator:
    """
    from C++ module
    create polynomial lane geometry from lane parameter from sensor signal BV1_LIN
    """
    def __init__(self):
        self.cubic_polynomial = Polynomial(3)
        self.discrete_sz = 0

    def createCartesianGeometry(self, linObj):
        discrete_points = []
        x = linObj['BeginnX']
        aa = 0.16666666666 * linObj['HorKreummAend']
        bb = 0.5 * linObj['HorKreumm']
        cc = linObj['GierWnkl']
        dd = linObj['AbstandY']

        self.cubic_polynomial.set_coeffs(0, -aa*x*x*x + bb*x*x - cc*x + dd)
        self.cubic_polynomial.set_coeffs(1, 3*aa*x*x - 2*bb*x + cc)
        self.cubic_polynomial.set_coeffs(2, -3*aa*x + bb)
        self.cubic_polynomial.set_coeffs(3, aa)
        # Appned the points
        while(x < linObj['EndeX'] and self.discrete_sz < kMaxNumberOfDiscretePoints):
            spherical_y = self.cubic_polynomial.evaluate(x)
            discrete_points.append((x, spherical_y))
            self.discrete_sz += 1
            x += kDistanceBetweenLinePoints
            if self.discrete_sz == kMaxNumberOfDiscretePoints:
                break
        # Append the last point
        self.discrete_sz += 1
        End_X = linObj['EndeX']
        End_y = self.cubic_polynomial.evaluate(End_X)
        discrete_points.append((End_X, End_y))

        return discrete_points
    
# for test ===================================================
def display(discrete_points):
    points_x = []
    points_y = []
    for point in discrete_points:
        points_x.append(point[0])
        points_y.append(point[1])
    plt.figure(figsize=(14, 4))
    plt.gca().set_aspect("equal")
    plt.plot(points_x, points_y)
    plt.show()

# for test ===================================================
if __name__ == "__main__":
    """
    ### BV1_LIN_01_ID = 6
    - Typ = 1
    - AbstandY = 1.55
    - BeginnX = 0
    - EndeX = 87.75
    - Breite = 0.3125
    - Krümm = -0.0002
    - Krümm' = -4.7683e-7
    """
    linObj = {}
    linObj['BeginnX'] = 0.0
    linObj['EndeX'] = 89.5
    linObj['AbstandY'] = -1.4609375
    linObj['HorKreumm'] = -0.0002288818
    linObj['HorKreummAend'] = -1.90734e-6
    linObj['GierWnkl'] = 0.0083

    laneGen = LaneGenerator()
    points_xy = laneGen.createCartesianGeometry(linObj)
    display(points_xy)
