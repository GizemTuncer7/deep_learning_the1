import math
from typing import Union
import math
import random
import numpy as np
import time

def deflatten(flat_list, dim):
    reversed_dim = dim[::-1]
    if dim == (0,):
        reversed_dim = (1, )
    for i in range(len(reversed_dim)):
        flat_list = [flat_list[j:reversed_dim[i] + j] for j in range(0, len(flat_list), reversed_dim[i])]
    return flat_list[0]


def flatten(data):
    flat_list = []
    if not isinstance(data, Union[list, tuple]):
        return [data]
    for item in data:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


class gergen:
    __veri = None
    D = None
    __boyut = None

    def __init__(self, veri=None):

        # veri can be a scalar
        if isinstance(veri, int) or isinstance(veri, float):
            self.__veri = veri
            self.__boyut = (0,)
            self.D = veri

        elif isinstance(veri, list):
            # veri can be a list of list
            if any(isinstance(ins, list) for ins in veri):
                self.__veri = veri
                self.__boyut = self.get_length(veri)
                self.D = self.calculate_transpose()
            # veri can be list
            else:
                self.__veri = veri
                self.__boyut = (len(veri), )
                self.D = self.calculate_transpose()
        # veri can be None
        else:
            self.__veri = []
            self.__boyut = (0,)
            self.D = []

    def __getitem__(self, index):
        """
        Returns the element of the gergen object at the specified index.
        :param index:
        :return:
        """
        tensor = gergen(self.listeye()[index])
        return tensor

    def __str__(self):
        """
        Returns a string representation of the gergen object.
        """
        boyut_size = len(self.boyut())
        return_string = ''

        if boyut_size == 1:
            if self.boyut() != (0, ):
                return_string += f'{self.boyut()[0]} boyutlu gergen :\n'
            else:
                return_string += f'{self.boyut()[0]} boyutlu skaler gergen :\n'

        else:
            return_string += ' x '.join(str(dim) for dim in self.boyut())
            return_string += f' boyutlu gergen :\n'

        return return_string + self.string_manipulation(str(self.listeye()))

    def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Multiplication operation for gergen objects.
        Called when a gergen object is multiplied by another, using the '*' operator.
        Could be element-wise multiplication or scalar multiplication, depending on the context.
        """
        if not isinstance(other, Union[gergen, int, float]):
            raise TypeError(f'{type(other)} is not supported for multiplication with gergen.')

        mult_tensor_data = None

        if isinstance(other, Union[int, float]):
            flat_list = flatten(self.listeye())
            mult_flat_list = [x * other for x in flat_list]
            mult_tensor_data = deflatten(mult_flat_list, self.boyut())

        elif isinstance(other, gergen):
            if isinstance(other.listeye(), Union[int, float]):
                flat_list = flatten(self.listeye())
                other_data = other.listeye()
                mult_flat_list = [x * other_data for x in flat_list]
                mult_tensor_data = deflatten(mult_flat_list, self.boyut())

            else:
                if self.boyut() == other.boyut():
                    flat_list = flatten(self.listeye())
                    other_flat_list = flatten(other.listeye())
                    mult_flat_list = [x * y for x, y in zip(flat_list, other_flat_list)]
                    mult_tensor_data = deflatten(mult_flat_list, self.boyut())
                else:
                    raise ValueError(f'Tensors with dimensions of {self.boyut()} and {other.boyut()} can not be multiplied.')

        new_tensor = gergen(mult_tensor_data)
        return new_tensor

    def __rmul__(self, other: Union['gergen', int, float]) -> 'gergen':
        return self.__mul__(other)

    def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Division operation for gergen objects.
        Called when a gergen object is divided by another, using the '/' operator.
        The operation is element-wise.
        """
        if other == 0:
            raise ZeroDivisionError('Division by zero is not allowed.')

        if isinstance(other, Union[int, float]):
            flat_list = flatten(self.listeye())
            div_flat_list = [x / other for x in flat_list]
            div_tensor_data = deflatten(div_flat_list, self.boyut())

        elif isinstance(other, gergen) and isinstance(other.listeye(), Union[int, float]):
            if other.listeye() == 0:
                raise ZeroDivisionError('Division by zero is not allowed.')
            flat_list = flatten(self.listeye())
            other_data = other.listeye()
            div_flat_list = [x / other_data for x in flat_list]
            div_tensor_data = deflatten(div_flat_list, self.boyut())

        else:
            raise TypeError(f'{type(other)} is not supported for division with gergen.')

        new_tensor = gergen(div_tensor_data)
        return new_tensor

    def __rtruediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        return self.us(-1) * other

    def __add__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Addition operation for gergen objects.
        """
        if not isinstance(other, Union[gergen, int, float]):
            raise TypeError(f'{type(other)} is not supported for addition with gergen.')

        added_tensor_data = None

        if isinstance(other, Union[int, float]):
            flat_list = flatten(self.listeye())
            added_flat_list = [x + other for x in flat_list]
            added_tensor_data = deflatten(added_flat_list, self.boyut())

        elif isinstance(other, gergen):
            if isinstance(other.listeye(), Union[int, float]):
                flat_list = flatten(self.listeye())
                other_data = other.listeye()
                added_flat_list = [x + other_data for x in flat_list]
                added_tensor_data = deflatten(added_flat_list, self.boyut())

            else:
                if self.boyut() == other.boyut():
                    flat_list = flatten(self.listeye())
                    other_flat_list = flatten(other.listeye())
                    added_flat_list = [x + y for x, y in zip(flat_list, other_flat_list)]
                    added_tensor_data = deflatten(added_flat_list, self.boyut())
                else:
                    raise ValueError(f'Tensors with dimensions of {self.boyut()} and {other.boyut()} can not be added.')

        new_tensor = gergen(added_tensor_data)
        return new_tensor

    def __radd__(self, other: Union['gergen', int, float]) -> 'gergen':
        return self.__add__(other)

    def __sub__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Subtraction operation for gergen objects.
        """
        if not isinstance(other, Union[gergen, int, float]):
            raise TypeError(f'{type(other)} is not supported for subtraction with gergen.')

        subtracted_tensor_data = None

        if isinstance(other, Union[int, float]):
            flat_list = flatten(self.listeye())
            subtracted_flat_list = [x - other for x in flat_list]
            subtracted_tensor_data = deflatten(subtracted_flat_list, self.boyut())

        elif isinstance(other, gergen):
            if isinstance(other.listeye(), Union[int, float]):
                flat_list = flatten(self.listeye())
                other_data = other.listeye()
                subtracted_flat_list = [x - other_data for x in flat_list]
                subtracted_tensor_data = deflatten(subtracted_flat_list, self.boyut())

            else:
                if self.boyut() == other.boyut():
                    flat_list = flatten(self.listeye())
                    other_flat_list = flatten(other.listeye())
                    subtracted_flat_list = [x - y for x, y in zip(flat_list, other_flat_list)]
                    subtracted_tensor_data = deflatten(subtracted_flat_list, self.boyut())
                else:
                    raise ValueError(f'Tensors with dimensions of {self.boyut()} and {other.boyut()} can not be subtracted.')

        new_tensor = gergen(subtracted_tensor_data)
        return new_tensor

    def __rsub__(self, other: Union['gergen', int, float]) -> 'gergen':
        return -1 * self + other

    def get_length(self, data, dim=None):
        if dim is None:
            dim = []

        if any(isinstance(ins, list) for ins in data):
            dim.append(len(data))
            return self.get_length(data[0], dim)
        else:
            dim.append(len(data))
            return tuple(dim)

    def string_manipulation(self, data):
        """
        Manipulates the data to make it more readable.
        :param data: data to be manipulated
        :return: manipulated data
        """

        replaced_data = data.replace('],', ']\n')
        return str(replaced_data)

    def uzunluk(self):
        """
        Returns the number of elements in the gergen object.
        """

        flattened_list = flatten(self.__veri)
        return len(flattened_list)

    def boyut(self):
        return self.__boyut

    def devrik(self):
        if len(self.boyut()) == (1, ) or len(self.boyut()) == (0, ):
            return self
        else:
            return gergen(self.D)

    def calculate_transpose(self):
        """
        calculates and returns the transpose of the gergen object.
        """
        old_flat_list = flatten(self.listeye())
        size = len(old_flat_list)
        transpose_flat_list = [0] * size
        for i in range(len(old_flat_list)):
            index_list = self.find_index(self.boyut(), i)
            transpose_index = self.find_transpose_index(index_list)
            transpose_flat_list[transpose_index] = old_flat_list[i]

        transpose_tensor_data = deflatten(transpose_flat_list, self.boyut()[::-1])
        return transpose_tensor_data

    def find_transpose_index(self, index_list):
        """
        Finds the transpose index of the given index list.
        """
        result = 0
        for i, value in enumerate(index_list[::-1]):
            result += (value * self.last_multiplication(self.boyut()[::-1], i))
        return result

    def find_index(self, dims, flat_index):
        """
        Finds the index of the given flat index.
        """
        dim_size = len(dims)
        to_be_updated_index = flat_index
        index_list = []
        for i in range(dim_size):
            last_multiplication_result = self.last_multiplication(dims, i)
            to_be_appended_index = to_be_updated_index // last_multiplication_result
            index_list.append(to_be_appended_index)
            to_be_updated_index = to_be_updated_index % last_multiplication_result
        return index_list

    def last_multiplication(self, to_be_multiplied_list, index_after):
        """
        Multiplies the elements of the list after the specified index.
        """
        mult = 1
        to_be_multiplied_list_iterated = to_be_multiplied_list[index_after + 1:]
        for i in to_be_multiplied_list_iterated:
            mult *= i
        return mult

    def sin(self):
        """
        Returns the sine of the gergen object.
        """
        flat_list = flatten(self.listeye())
        sin_tensor_data = deflatten([math.sin(x) for x in flat_list], self.boyut())
        new_tensor = gergen(sin_tensor_data)
        return new_tensor

    def cos(self):
        """
        Returns the cosine of the gergen object.
        """
        flat_list = flatten(self.listeye())
        sin_tensor_data = deflatten([math.cos(x) for x in flat_list], self.boyut())
        new_tensor = gergen(sin_tensor_data)
        return new_tensor

    def tan(self):
        """
        Returns the tangent of the gergen object.
        """
        flat_list = flatten(self.listeye())
        sin_tensor_data = deflatten([math.tan(x) for x in flat_list], self.boyut())
        new_tensor = gergen(sin_tensor_data)
        return new_tensor

    def us(self, n: int):
        """
        Returns the gergen object raised to the power of n.
        """

        if n < 0:
            raise ValueError(f'{n} should be integer.')

        flat_list = flatten(self.listeye())
        powered_tensor_data = deflatten([math.pow(x, n) for x in flat_list], self.boyut())
        new_tensor = gergen(powered_tensor_data)
        return new_tensor

    def log(self):
        flat_list = flatten(self.listeye())
        log_tensor_data = deflatten([math.log10(x) for x in flat_list], self.boyut())
        new_tensor = gergen(log_tensor_data)
        return new_tensor

    def ln(self):
        flat_list = flatten(self.listeye())
        ln_tensor_data = deflatten([math.log(x) for x in flat_list], self.boyut())
        new_tensor = gergen(ln_tensor_data)
        return new_tensor

    def L1(self):
        flat_list = flatten(self.listeye())
        return sum(flat_list)

    def L2(self):
        flat_list = flatten(self.listeye())
        squared_flat_list = [math.pow(x, 2) for x in flat_list]
        sum_of_squared_flat_list = sum(squared_flat_list)
        square_root = math.sqrt(sum_of_squared_flat_list)
        return square_root

    def Lp(self, p):
        if p < 0:
            raise ValueError(f'{p} should be positive.')

        flat_list = flatten(self.listeye())
        powered_flat_list = [math.pow(x, p) for x in flat_list]
        sum_of_powered_flat_list = sum(powered_flat_list)
        nth_root = math.pow(sum_of_powered_flat_list, 1 / p)
        return nth_root

    def listeye(self):
        return self.__veri

    def duzlestir(self):
        flat_data = flatten(self.listeye())
        new_gergen = gergen(flat_data)
        return new_gergen

    def boyutlandir(self, yeni_boyut):
        if not isinstance(yeni_boyut, tuple):
            raise TypeError(f'{type(yeni_boyut)} is not supported for reshaping the tensor. '
                            f'Expected boyut type is tuple.')

        flat_list = flatten(self.listeye())
        total_number_of_elem = len(flat_list)
        yeni_boyut_total_number_of_elem = 1

        for dim in yeni_boyut:
            yeni_boyut_total_number_of_elem *= dim

        if total_number_of_elem != yeni_boyut_total_number_of_elem:
            raise ValueError(f'Total number of elements in the tensor should be equal to the total number of elements '
                             f'in the new shape. Expected total number of elements is '
                             f'{yeni_boyut_total_number_of_elem} but got {total_number_of_elem}.')

        deflatten_data = deflatten(flat_list, yeni_boyut)
        new_gergen = gergen(deflatten_data)
        return new_gergen

    def ic_carpim(self, other):
        total = 0
        if not isinstance(other, gergen):
            raise TypeError('Both operands should be tensors.')

        if len(self.boyut()) == 1:
            if self.boyut() == other.boyut():
                flat_list = flatten(self.listeye())
                other_flat_list = flatten(other.listeye())
                for i in range(len(flat_list)):
                    total += (flat_list[i] * other_flat_list[i])
                return total
            else:
                raise ValueError('Tensors with different dimensions can not be multiplied.')

        elif len(self.boyut()) == 2:
            if self.boyut()[1] == other.boyut()[0]:
                matrix_a = self.listeye()
                matrix_b = other.listeye()

                result = [[0 for _ in range(other.boyut()[1])] for _ in range(self.boyut()[0])]

                for i in range(self.boyut()[0]):
                    for j in range(other.boyut()[1]):
                        for k in range(other.boyut()[0]):
                            result[i][j] += matrix_a[i][k] * matrix_b[k][j]
                new_tensor = gergen(result)
                return new_tensor

            else:
                raise ValueError('First tensor\'s number of column and second tensor\'s row number should be match.')
        else:
            raise ValueError('This operation is not supported for tensors with more than 2 dimensions.')

    def dis_carpim(self, other):
        if isinstance(other, gergen):
            if len(self.boyut()) == 1 and len(other.boyut()) == 1:
                flat_list = flatten(self.listeye())
                other_flat_list = flatten(other.listeye())
                outer_list = []
                for x in flat_list:
                    inner_list = []
                    for y in other_flat_list:
                        inner_list.append(x * y)
                    outer_list.append(inner_list)
                tensor_data = deflatten(outer_list, (self.boyut()[0], other.boyut()[0]))
                new_tensor = gergen(tensor_data)
                return new_tensor
            else:
                raise ValueError('This operation is not supported for tensors with more than 1 dimensions.')
        else:
            raise TypeError('Both operands should be tensors.')

    def topla(self, eksen=None):
        if eksen is not None and not isinstance(eksen, int):
            raise ValueError(f'Eksen must be None or integer. However; the given eksen is {eksen}.')

        if eksen is not None and (eksen >= len(self.boyut()) or eksen < 0):
            raise ValueError(f'Given eksen is out of range. The minimum eksen value is 0 and '
                             f'the maximum eksen value is {len(self.boyut()) - 1}.')

        if eksen is None:
            flat_list = flatten(self.listeye())
            return sum(flat_list)

        elif eksen == 0:
            current_data = self.listeye()
            tensor_data = self.rec_topla_secondary(current_data, eksen)
            new_tensor = gergen(tensor_data)
            return new_tensor

        else:
            tensor_data = self.rec_topla(self.listeye(), eksen)
            new_tensor = gergen(tensor_data)
            return new_tensor

    def rec_topla_secondary(self, current_data, eksen=None):
        if isinstance(current_data, list):
            if any(isinstance(ins, list) for ins in current_data):
                zipped_list = zip(*current_data)
                zipped_data = []
                for i in zipped_list:
                    zipped_data.append(i)
                liste_appended = []
                for x in zipped_data:
                    liste_appended.append(self.rec_topla_secondary(list(x), eksen))
                return liste_appended
            else:
                return sum(current_data)

    def rec_topla(self, current_data, eksen=None, depth=0):
        if eksen == depth:
            if isinstance(current_data, Union[list, tuple]):
                if any(isinstance(ins, Union[int, float]) for ins in current_data):
                    return sum(current_data)
                elif any(isinstance(ins, Union[list, tuple]) for ins in current_data[0]):
                    return_list = []
                    for i in zip(*current_data):
                        return_list.append(self.rec_topla(i, eksen, depth))
                    return return_list
                elif any(isinstance(ins, Union[list, tuple]) for ins in current_data):
                    zipped_data = zip(*current_data)
                    summed = []
                    for i in zipped_data:
                        summed.append(sum(i))
                    return summed
                else:
                    return sum(current_data)

        else:
            added_data = []
            for data in current_data:
                added_data.append(self.rec_topla(data, eksen, depth + 1))
            return added_data

    def ortalama(self, eksen=None):
        if eksen is not None and not isinstance(eksen, int):
            raise ValueError(f'Eksen must be None or integer. However; the given eksen is {eksen}.')

        if eksen is not None and(eksen >= len(self.boyut()) or eksen < 0):
            raise ValueError(f'Given eksen is out of range. The minimum eksen value is 0 and '
                             f'the maximum eksen value is {len(self.boyut()) - 1}.')
        if eksen is None:
            flat_list = flatten(self.listeye())
            return sum(flat_list) / len(flat_list)

        elif eksen == 0:
            current_data = self.listeye()
            tensor_data = self.rec_ortalama_secondary(current_data, eksen)
            new_tensor = gergen(tensor_data)
            return new_tensor

        else:
            tensor_data = self.rec_ortalama(self.listeye(), eksen)
            new_tensor = gergen(tensor_data)
            return new_tensor

    def rec_ortalama_secondary(self, current_data, eksen=None):
        if isinstance(current_data, list):
            if any(isinstance(ins, list) for ins in current_data):
                zipped_data = []
                zipped_list = zip(*current_data)
                for i in zipped_list:
                    zipped_data.append(i)
                liste_appended = []
                for x in zipped_data:
                    liste_appended.append(self.rec_ortalama_secondary(list(x), eksen))
                return liste_appended
            else:
                return sum(current_data) / len(current_data)

    def rec_ortalama(self, current_data, eksen=None, depth=0):
        if eksen == depth:
            if isinstance(current_data, Union[list, tuple]):
                if any(isinstance(ins, Union[int, float]) for ins in current_data):
                    return sum(current_data) / len(current_data)
                elif any(isinstance(ins, Union[list, tuple]) for ins in current_data[0]):
                    return_list = []
                    for i in zip(*current_data):
                        return_list.append(self.rec_ortalama(i, eksen, depth))
                    return return_list
                elif any(isinstance(ins, Union[list, tuple]) for ins in current_data):
                    zipped_data = zip(*current_data)
                    summed = []
                    for i in zipped_data:
                        summed.append(sum(i) / len(i))
                    return summed
                else:
                    return sum(current_data)/len(current_data)

        else:
            added_data = []
            for data in current_data:
                added_data.append(self.rec_ortalama(data, eksen, depth + 1))
            return added_data


def cekirdek(sayi: int):
    random.seed(sayi)


def rastgele_dogal(boyut, aralik=(0, 100), dagilim='uniform'):

    """
    Generates data of specified dimensions with random integer values and returns a gergen object.

    Parameters:
    boyut (tuple): Shape of the desired data.
    aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to (0,100), which implies a default range.
    dagilim (string, optional): Distribution of random values ('uniform'). Defaults to 'uniform'.

    Returns:
    gergen: A new gergen object with random integer values.
    """

    if dagilim != 'uniform':
        raise ValueError(f'{dagilim} distribution is not supported. This distribution should be uniform')

    total_number = 1
    for x in boyut:
        total_number *= x

    random_integer_list = [random.randint(aralik[0], aralik[1]) for x in range(total_number)]
    random_integer_data = deflatten(random_integer_list, boyut)
    new_tensor = gergen(random_integer_data)
    return new_tensor


def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
    """
    Generates a gergen of specified dimensions with random floating-point values.

    Parameters:
    boyut (tuple): Shape of the desired gergen.
    aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to (0.0, 1.0) for uniform distribution.
    dagilim (string, optional): Distribution of random value ('uniform'). Defaults to 'uniform'.

    Returns:
    gergen: A new gergen object with random floating-point values.
    """

    if dagilim != 'uniform':
        raise ValueError(f'{dagilim} distribution is not supported. This distribution should be uniform.')

    total_number = 1
    for x in boyut:
        total_number *= x

    random_integer_list = [random.uniform(aralik[0], aralik[1]) for x in range(total_number)]
    random_integer_data = deflatten(random_integer_list, boyut)
    new_tensor = gergen(random_integer_data)
    return new_tensor


"""
    Example Results
    example_1: 
        They are same
        Time taken for gergen: 0.15154647827148438
        Time taken for numpy: 0.003280162811279297
    example_2: 
        They are same
        Time taken for gergen: 1.3719775676727295
        Time taken for numpy: 0.00011992454528808594
    example_3:
        They are same
        Time taken for gergen: 0.570544958114624
        Time taken for numpy: 0.00035834312438964844
"""
def are_the_same(tensor, numpy_arr):
    flat_tensor_list = flatten(tensor.listeye())
    flat_numpy_list = numpy_arr.flatten()
    for i in range(len(flat_tensor_list)):
        if abs(flat_tensor_list[i] - flat_numpy_list[i]) > 0.0001:
            return False
    return True

def example_1():
    # Example 1
    boyut = (64, 64)
    g1 = rastgele_gercek(boyut)
    g2 = rastgele_gercek(boyut)
    np_1 = np.array(g1.listeye())
    np_2 = np.array(g2.listeye())

    start = time.time()
    result_g = g1.ic_carpim(g2)
    end = time.time()

    start_np = time.time()
    result_np = np_1.dot(np_2)
    end_np = time.time()

    if are_the_same(result_g, result_np):
        print("They are same")
    else:
        print("They are different")

    print("Time taken for gergen:", end - start)
    print("Time taken for numpy:", end_np - start_np)


def example_2():
    gergen_a = rastgele_gercek((4, 16, 16, 16))
    gergen_b = rastgele_gercek((4, 16, 16, 16))
    gergen_c = rastgele_gercek((4, 16, 16, 16))
    start = time.time()
    result = (gergen_a * gergen_b + gergen_a * gergen_c + gergen_b * gergen_c).ortalama()
    end = time.time()

    np_a = np.array(gergen_a.listeye())
    np_b = np.array(gergen_b.listeye())
    np_c = np.array(gergen_c.listeye())
    start_np = time.time()
    result_np = (np_a * np_b + np_a * np_c + np_b * np_c).mean()
    end_np = time.time()

    epsi = 0.0001
    if abs(result - result_np) < epsi:
        print("They are same")
    else:
        print("They are different")

    print("Time taken for gergen:", end - start)
    print("Time taken for numpy:", end_np - start_np)


def example_3():
    gergen_a = rastgele_gercek((3, 64, 64))
    gergen_b = rastgele_gercek((3, 64, 64))

    start = time.time()
    result = (gergen_a.sin() + gergen_b.cos().us(2)).ln() / 8
    end = time.time()

    np_a = np.array(gergen_a.listeye())
    np_b = np.array(gergen_b.listeye())

    start_np = time.time()
    result_np = np.log(np.sin(np_a) + np.cos(np_b) ** 2) / 8
    end_np = time.time()

    if are_the_same(result, result_np):
        print("They are same")
    else:
        print("They are different")

    print("Time taken for gergen:", end - start)
    print("Time taken for numpy:", end_np - start_np)
