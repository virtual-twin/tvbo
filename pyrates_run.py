from numpy import pi, sqrt
from numpy import dot
from numpy import roll


def vector_field(t,y,dy,gamma_par,local_coupling,Ipar,f,alpha,e,g,tau,d,beta_par,c,b,a,V_buffer,source_idx,V_delays,weight):


	V = y[0:300]
	W = y[300:600]
	V_buffer[:] = roll(V_buffer, 1, 1)
	V_buffer[:,0] = V
	V_buffered = V_buffer[source_idx,V_delays]
	c_glob = dot(weight, V_buffered)
	
	dy[0:300] = d*tau*(Ipar*gamma_par - V**3*f + V**2*e + V*g + V*local_coupling + W*alpha + c_glob*gamma_par)
	dy[300:600] = d*(V**2*c + V*b - W*beta_par + a)/tau

	return dy