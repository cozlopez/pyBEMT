 
# fixing folder imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './..')))


import math



def get_ISA(alt: float) -> tuple:
    '''This function calculates the ISA conditions based on the measured altitude.
    Inputs:
        alt: altidude in feet
    Outputs:
        P: Pressure in Pa
        rho: Density in kg/m^3
        tempK: Statis temperature in K'''

    alt  = alt*0.3048   # convert feet to meters

    # some reference values
    refalt = 0
    reftemp = 288.15
    refP = 101325

    list1 = [11000, 20000, 32000, 47000, 51000,71000,86000]     # list of altitude ranges
    list2 = [-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002]     # list of corresponding rate of temperature change

    # iterating through the lists to find the final values for the given altitude
    for element1, element2 in zip(list1, list2):

        # calculates the pressure if there is no rate of temperature change
        if element2 == 0:
            P = refP * math.e ** ((-9.81 / (287 * tempK)) * (min(alt, element1) - refalt))
            refalt = element1
            refP = P    # updating the reference pressure for the next iteration

            # checks if the altitude is less than the current element1, if it is then it breaks out of the loop
            if alt <= element1:
                break
            
        # calculates the pressure and temperature based on the rate of temperature change
        else:
            tempK = reftemp + element2 * (min(alt, element1) - refalt)
            P = refP * (tempK / reftemp) ** (-9.80665 / (element2 * 287))
            reftemp = tempK     # updating the reference temperature for the next iteration
            refP = P            # updating the reference pressure for the next iteration
            refalt = element1

            # checks if the altitude is less than the current element1, if it is then it breaks out of the loop
            if alt <= element1:
                break

    # calculating the density using the ideal gas law
    rho = P / (287 * tempK)
    gamma = 1.4
    R = 287
    speed_of_sound = math.sqrt(gamma * R * tempK)
    
    return P, rho, tempK, speed_of_sound
