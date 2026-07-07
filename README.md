# GIDC and L1-Magic
Tensorflow implementation of paper: [Far-field super-resolution ghost imaging with a deep neural network constraint](https://www.nature.com/articles/s41377-021-00680-w). One of the experiment data was provided.

L1-Magic Implementation: [Hadamard single-pixel imaging versus Fourier single-pixel imaging](https://opg.optica.org/oe/fulltext.cfm?uri=oe-25-16-19619).


## Information
**1. Data :** This contains data. 
**i. For SP(in phase) :** Save patterns using **pattern_save_code** and using **linking** you can link measurements i.e. **0, 2pibi3,    4pibi3** to make data.
**ii. For DHSI :** You can directly save data using **DHI**.
**iii. For CGI :** You can directly save data using **GI_speckle_data**.

**2. Object :** This contains figures, which is required in the code.

**3. Python_Codes :** This contains required codes.You can check for differnet dimensions and differnt number of patterns by  changing num_pattern in DHI and CGI.
**i. For SP(in phase) :** Use code **GIDC_main**.
**ii. For DHSI :** Use code **DHI_32**.
**iii. For CGI :** Use code **CGI_1000_64**.

**4. L1-Magic :** This contains matlab software along with codes for DHSI(4codes) and FSI(**FSI.m**).

## How to use
**Step 1: Configuring required packages**

python 3.9

tensorflow 2.10.0

matplotlib 3.9.4

numpy 1.23.5

pillow 11.3.0

scipy 1.13.1

h5py  3.14.0

contourpy 1.3.0

cycler 0.12.1 

fonttools 4.60.2 

kiwisolver 1.4.7 

pyparsing 3.2.5 

python-dateutil 2.9.0.post0 

importlib-resources 6.5.2

**Step 2: Run GIDC_main.py after download and extract the ZIP file.**

# Copyright
Copyright © Indian Institute of Technology (BHU), Varanasi.

This software was developed by Jahnavi Gupta under the guidance of Dr. Rakesh K. Singh (Professor, Department of Physics, IIT BHU, Varanasi) as part of an Integrated Dual Degree (B.Tech. + M.Tech.) thesis in Engineering Physics at IIT (BHU), Varanasi.

The author is permitted to reproduce and authorize reproduction of derivative works, provided that the source and the Institute's copyright notice are indicated.
