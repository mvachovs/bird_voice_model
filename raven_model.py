import math
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter, freqz
import numpy

#Model of raven syrinx and trachea as defined by Fletcher in Bird Song: A Quantitative Acoustic Model
#and by Julius O. Smith in The Sounds of the Avian Syrinx - Are they really Flute-like?
#The model integrates three differential equations using the trapezoid method to determine the value
#of the pressure in the bronchus (p0), the displacement of the syringeal membrane (x), and the volume
#airflow from the syrinx (U). The pressure in the trachea (p1) is then calculated using delaylines and
#a low pass filter.
class fletcher_bird_voice():

	def __init__(self, trachea_radius, trachea_length, membrane_radius, membrane_thickness, membrane_density, membrane_nonlinear, equilibrium_opening, bronchial_volume, airsac_pressure, air_density, speed_of_sound, period, num_modes, modes):
		#initialize to 0
		self.p0, self.p0prime, self.p1, self.U, self.Uprime, self.x = (0,)*6
		self.x_split = [0]*num_modes
		self.xprime = [0]*num_modes
		self.xdoubleprime = [0]*num_modes

		#constants
		self.mode_freq = modes
		self.ang_mode_freq = []
		for i in range(num_modes):
			self.ang_mode_freq.append(2*math.pi*modes[i])
		self.num_modes = num_modes
		self.a = trachea_radius
		self.h = membrane_radius
		self.d = membrane_thickness
		self.rom = membrane_density
		self.n = membrane_nonlinear
		self.x0 = equilibrium_opening
		self.V = bronchial_volume
		self.pg = airsac_pressure
		self.ro = air_density
		self.c = speed_of_sound
		self.Z0 = (self.ro*self.c)/(math.pi*math.pow(self.a,2))
		self.Zg = self.Z0*(1/math.pow(5,2))*math.pow(self.a,2)
		self.m0 = self.rom*self.a*self.h*self.d*math.pi/4
		self.m = self.m0
		self.T = period

		#for trachea
		self.L = trachea_length
		self.sample_length = int(trachea_length/(self.c*self.T))
		self.sample_length = 7
		self.output = 0
		self.upperline = [0]*self.sample_length
		#self.upperline = [0,.1,.2,.3,.4,.5]
		self.lowerline = [0]*self.sample_length
		self.filtered_lowerline = [0]*self.sample_length
		self. upper_index, self.lower_index = 0, 0

		#for plotting
		self.toplotp0, self.toplotx, self.toplotU, self.toplotp1, self.toplotoutput, self.toplotUprime, self.toplotp0prime = [],[],[],[],[],[],[]
		self.toplotmodes = []
		for i in range(num_modes):
			self.toplotmodes.append([])

	#Runs Fletcher model for given number of iterations. If with_trachea is true, includes reflections from the trachea/the pressure
	#inside the trachea in the calculations, if false, ignores the pressure inside the trachea.
	def speak(self, num_iter, with_trachea):
		for i in range(num_iter):
			#Update bronchial pressure
			self.updatep0()
			self.toplotp0.append(self.p0)
			self.toplotp0prime.append(self.p0prime)

			#Update syringeal displacement
			self.updatex()
			self.toplotx.append(self.x)
			for i in range(self.num_modes):
				self.toplotmodes[i].append(self.x_split[i])

			#Update volume flow out of syrinx
			self.updateU()
			self.toplotU.append(self.U)		
			self.toplotUprime.append(self.Uprime)
			
			#Update tracheal output and pressure in trachea
			if(with_trachea):
				self.reflect()
				self.toplotoutput.append(self.output)
				self.toplotp1.append(self.p1)

			print("p0: " + str(self.p0))
			print("x: " + str(self.x))
			for i in range(self.num_modes):
				print("x" + str(i + 1) + ": " + str(self.x_split[0]))
			print("U: " + str(self.U))
			print("p1: " + str(self.p1))
			print("upper index: " + str(self.upper_index))
			print("lower index: " + str(self.lower_index))
			print("upper line: " + str(self.upperline))
			print("lower line: " + str(self.lowerline))
			print("filtered lower line " + str(self.filtered_lowerline) + "\n")
			print("p0prime: " + str(self.p0prime))
			print("Uprime: " + str(self.Uprime))
			for i in range(self.num_modes):
				print("x" + str(i + 1) + "prime: " + str(self.xprime[i]))
				print("x" + str(i + 1) + "doubleprime: " + str(self.xdoubleprime[i]))
			print("\n")

	#Plot relevant values
	def plot_speech(self):
		plt.plot(self.toplotp0, label="Bronchial Pressure (kPa)")
		#plt.plot(self.toplotp1, label="Tracheal Pressure (kPa)")
		plt.plot(self.toplotx, label="Total Displacement (mm)")
		#plt.plot(self.toplotU, label="Volume Flow (mm^3/s)")
		#plt.plot(self.toplotp0prime, label="Bronchial Pressure Prime")
		#plt.plot(self.toplotUprime, label="Volume Flow Prime")
		plt.plot(self.toplotoutput, label="Output from Trachea (kPa)")
		#for i in range(self.num_modes):
		#	plt.plot(self.toplotmodes[i], label="Individual Displacement (mm)")
		plt.show()

	#Write the tracheal output to a sound file
	def output_file(self):
		wavfile.write("sound_output.wav", 32000, numpy.asarray(self.toplotoutput))

	#Update the bronchial pressure
	def updatep0(self):

		acoustic_stiffness = self.ro*math.pow(self.c,2)/self.V
		volume_flow_in = (self.pg - self.p0)/self.Zg
		newp0prime = acoustic_stiffness*(volume_flow_in - self.U)
		self.p0 = integrate(self.p0, self.p0prime, newp0prime, self.T)
		self.p0prime = newp0prime

	#Update the tracheal pressure (taken at the bottom of the trachea)
	#Tracheal pressure is the reflections of pressure from the top
	#of the trachea (lower) + the airflow times tracheal impedance.
	def updatep1(self, lower):

		alpha = .000012*math.sqrt(self.ang_mode_freq[0])/self.a
		B = 1 - (2*alpha*self.L)
		self.p1 = (self.Z0*self.U) + lower*B

	#Update the airflow out of the syrinx. Airflow is 0 when the syrinx
	#is closed.
	def updateU(self):
		if self.x != 0:
			C = self.ro/(8*math.pow(self.a,2)*math.pow(self.x,2))
			if self.x < self.a:
				D = 2*math.sqrt(self.a*self.x)/self.ro
			else: #to follow bounds (pg478 in fletcher)
				D = 1 
			newUprime = D*(self.p0 - self.p1 - (C*math.pow(self.U,2)))
			self.U = integrate(self.U, self.Uprime, newUprime, self.T)
			self.Uprime = newUprime
			#remove neg spikes that occur when x is too small
			if self.U < 0:
				self.Uprime = 0
				self.U = 0

		else:
			self.Uprime = 0
			self.U = 0

	#Update displacement, which is the addition of the displacement
	#of individual modes. Cannot be negative.
	def updatex(self):
		curx = 0
		for i in range(self.num_modes):
			curx += self.update_single_x(i)
		if curx < 0:
			self.x = 0
		else:
			self.x = curx

	#Update the displacement of a single mode. Integrates twice, as the differential
	#equation calculates x".
	def update_single_x(self, i):

		k = self.ang_mode_freq[i]/(2*self.get_Q(self.mode_freq[i]))
		if self.x == 0:
			k = k*10
		
		#if syrinx is closed, then tracheal pressure does not affect force (unsure - not specified in model)
		F = (self.p0)/2
		if self.x > 0:
			F = (self.p0 + self.p1)/2
			F -= (self.ro*math.pow(self.U,2))/(7*math.sqrt(self.a*math.pow(self.x,3)))
		F = F*2*self.a*self.h
		m = self.calcmass(i)

		newxdoubleprime = (F/m) - (2*k*self.xprime[i]) - (math.pow(self.ang_mode_freq[i],2)*(self.x_split[i] - self.x0))
		newxprime = integrate(self.xprime[i], self.xdoubleprime[i], newxdoubleprime, self.T)
		self.xdoubleprime[i] = newxdoubleprime
		self.x_split[i] = integrate(self.x_split[i], self.xprime[i], newxprime, self.T)
		self.xprime[i] = newxprime

		return self.x_split[i]

	#Reflects the pressure from the trachea and updates the tracheal pressure accordingly.
	#Uses a delay-line with sample length corresponding to the length of the trachea to
	#calculate the pressure at the top of the trachea (i.e. the output), filters the output
	#using a low pass filter, and then uses a second delay-line of the same length to add the
	#pressure from reflection to the tracheal pressure p1.
	def reflect(self):
		if(self.lower_index == 0):
			#LPF
			self.filtered_lowerline = self.butter_lowpass_filter(self.lowerline, 9000, 1/self.T, 6)
		self.updatep1(self.filtered_lowerline[self.lower_index])

		#update delay lines
		upper, self.upper_index = self.delayline(self.upperline, self.upper_index, self.p1)
		lower, self.lower_index = self.delayline(self.lowerline, self.lower_index, -upper)

		self.output = upper

	#Returns the Q value for a given freq
	def get_Q(self, freq):

		return freq/75

	#Calculates the effective mass, which increases as the syrinx vibrates (as x becomes nonzero),
	#and surrounding tissue take part in the vibration.
	def calcmass(self,i):	
		if self.x_split[i] > 0:	
			difference = (self.x_split[i] - self.x0)/self.h
			return self.m0*(1 + (self.n*math.pow(difference,2)))
		return self.m0

	#Delayline as buffer
	def delayline(self,line,index,value):
		y = line[index]
		line[index] = value
		index += 1
		if index >= self.sample_length:
			index -= self.sample_length
		return y, index

	#Low pass filter
	def butter_lowpass(self, cutoff, fs, order=5):
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = butter(order, normal_cutoff, btype='low', analog=False)
		return b, a

	#Low pass filter
	def butter_lowpass_filter(self, data, cutoff, fs, order=5):
		b, a = self.butter_lowpass(cutoff, fs, order=order)
		y = lfilter(b, a, data)
		return y

class gardner_syrinx():

	def __init__(self,upper_labia,lower_labia,t_constant,mass,restitution_constant,D_coefficient,D2_coefficient,bronchial_pressure,period,parameter_freq):

		#initialize
		self.x = 0
		self.xprime = 0
		self.xdoubleprime = 0
		self.toplotx, self.toplotK, self.toplotPb, self.toplotxprime, self.toplotxdoubleprime, self.toplotPf = [], [], [], [], [], []
		self.K = 0
		self.Pb = 0

		#constants
		self.a0 = upper_labia
		self.b0 = lower_labia
		self.t = t_constant
		self.M = mass
		self.K_scale = restitution_constant
		self.D = D_coefficient
		self.D2 = D2_coefficient
		self.Pb_scale = bronchial_pressure
		self.T = period
		self.freq = parameter_freq


	def speak(self, num_iter):
		for i in range(num_iter):
			self.oscillate(i)
			self.toplotx.append(self.x)
			self.toplotxprime.append(self.xprime)
			self.toplotxdoubleprime.append(self.xdoubleprime)
			print("x: " + str(self.x))
			print("xprime: " + str(self.xprime))
			print("xdoubleprime: " + str(self.xdoubleprime) + "\n")

	def plot_speech(self):
		plt.plot(self.toplotx)
		#plt.plot(self.toplotK)
		#plt.plot(self.toplotPb)
		plt.plot(self.toplotPf)
		#plt.plot(self.toplotxprime)
		#plt.plot(self.toplotxdoubleprime)
		plt.show()

	def oscillate(self,i):
		newxdoubleprime = self.calc_xdoubleprime(i)
		newxprime = integrate(self.xprime, self.xdoubleprime, newxdoubleprime, self.T)
		self.xdoubleprime = newxdoubleprime
		self.x = integrate(self.x, self.xprime, newxprime, self.T)
		self.xprime = newxprime
		#if i < 75:
		#	self.K = 0
		#else:
		#	self.K = self.K_scale*math.cos((2*math.pi*self.T*75*(i-75)) + math.pi) + self.K_scale
		self.K = self.K_scale*math.sin((2*math.pi*self.T*self.freq*i)) + self.K_scale
		self.toplotK.append(self.K*.000001)
		self.Pb = self.Pb_scale*math.cos((2*math.pi*self.T*self.freq*i) + math.pi) + self.Pb_scale
		self.toplotPb.append(self.Pb*.00001)

	def calc_xdoubleprime(self,i):
		a = self.a0 + self.x + (self.t*self.xprime)
		b = self.b0 + self.x - (self.t*self.xprime)
		Pf = self.Pb*(1 - (a/b))
		self.toplotPf.append(Pf)
		#print(a)
		#print(b)
		#print(Pf)
		#print(self.K*self.x)
		#print(self.D2*math.pow(self.xprime,3))
		#print(self.D*self.xprime)
		#return (Pf - (self.K*self.x) - (self.D*self.xprime))/self.M
		return (Pf - (self.K*self.x) - (self.D2*math.pow(self.xprime,3)) - (self.D*self.xprime))/self.M

def integrate(tointegrate, old_value, new_value, period):

	return tointegrate + (old_value + new_value)*(period/2)

def test_fletcher():

	trachea_radius = 3.5 #mm
	trachea_length = 70 #mm
	membrane_radius = 3.5 #mm
	membrane_thickness = .1 #mm
	membrane_density = .000001 #kg/mm^3
	membrane_nonlinear = 10 #constant
	equilibrium_opening = 0 #mm
	bronchial_volume = 1000 #mm^3
	airsac_pressure = .3 #kg/(mm*s^2)
	air_density = .000000001225 #kg/mm^3
	speed_of_sound = 340270 #mm/s at 15 degrees C
	period = 1/32000 #s
	modes = [150,250] #Hz (1/s)

	raven = fletcher_bird_voice(trachea_radius, trachea_length, membrane_radius, membrane_thickness, membrane_density, membrane_nonlinear, equilibrium_opening, bronchial_volume, airsac_pressure, air_density, speed_of_sound, period, 2, modes)
	raven.speak(1000, True)
	raven.plot_speech()
	raven.output_file()

def test_gardner():

	fs = 32000       # sampling rate, Hz, must be integer
	duration = 1.0   # in seconds, may be float
	f = 1000.0       # sine frequency, Hz, may be float

	#bronchial_pressure = (numpy.sin(2*numpy.pi*numpy.arange(fs*duration)*f/fs)).astype(numpy.float32)
	#restitution_constant = (numpy.sin(2*numpy.pi*numpy.arange(fs*duration)*f/fs)).astype(numpy.float32)

	upper_labia = .02 #cm
	lower_labia = .04 #cm
	t_constant = .00015 #s
	mass = .005 #g/cm3
	restitution_constant = 200000.0 #g*cm/s2cm3
	D_coefficient = 5.0 #dynes*s/cm3
	D2_coefficient = .01 #dyne*s/cm5
	bronchial_pressure = 10000.0 #g/(cm*s2)
	period = 1/32000.0
	parameter_freq = 200.0
	canary = gardner_syrinx(upper_labia,lower_labia,t_constant,mass,restitution_constant,D_coefficient,D2_coefficient,bronchial_pressure,period, parameter_freq)
	canary.speak(32000)
	canary.plot_speech()

#test_gardner()
test_fletcher()