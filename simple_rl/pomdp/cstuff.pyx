import random
cpdef sample_state(list b):
	cdef double cumulative_probability = 0
	cdef int i
	for i in range(len(b)):
		cumulative_probability += b[i]
		if random.random() < cumulative_probability:
			return i
	# In case the distribution added to slightly below 1 and we had bad luck
	return random.sample([i for i in range(len(b))],1)[0]