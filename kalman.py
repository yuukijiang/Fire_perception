import numpy as np
import matplotlib.pyplot as plt
import math


class kalman:
    def __init__(self, Q, R):
        self.Q = Q
        self.R = R

        self.P_k_k1 = 1
        self.Kg = 0
        self.P_k1_k1 = 1
        self.x_k_k1 = 0
        self.ADC_OLD_Value = 0
        self.Z_k = 0
        self.kalman_adc_old = 0

    def kalman_prepare(self, ADC_Value,aaa = 60):

        self.Z_k = ADC_Value

        if (abs(self.kalman_adc_old - ADC_Value) >= aaa):
            self.x_k1_k1 = ADC_Value * 0.382 + self.kalman_adc_old * 0.618
        else:
            self.x_k1_k1 = self.kalman_adc_old;

        self.x_k_k1 = self.x_k1_k1
        self.P_k_k1 = self.P_k1_k1 + self.Q

        self.Kg = self.P_k_k1 / (self.P_k_k1 + self.R)

        kalman_adc = self.x_k_k1 + self.Kg * (self.Z_k - self.kalman_adc_old)
        self.P_k1_k1 = (1 - self.Kg) * self.P_k_k1
        self.P_k_k1 = self.P_k1_k1

        self.kalman_adc_old = kalman_adc

        return kalman_adc

    def kalman_filter(self,ADC,aaa):
        self.adc = []
        for item in ADC:
            self.adc.append(kalman.kalman_prepare(self, item, aaa))
        return self.adc
if __name__ == '__main__':

    a = [100] * 200
    array = np.array(a)

    s = np.random.normal(0, 15, 200)
    test_array = array + s
    li = kalman(0.001, 0.1)
    adc = li.kalman_filter(test_array,60)
    plt.plot(adc)
    plt.plot(array)
    plt.plot(test_array)
    plt.show()