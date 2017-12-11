from __future__ import print_function
import numpy as np
import cv2
import time
import os
import imagehash
from PIL import Image
from collections import defaultdict

#Here we check the result, this is a seperate function as 
# sklearn and multiprocessing.set_start_method('spawn')
# were not functioning together in the same scope
def check_res(result_sets):
	import adjusted_rand_index
	#We just simply run the script given by the teacher and print the results
	print(adjusted_rand_index.rand_index(result_sets))

#Here we do the clustering, same reason as above for why
# this is a seperate function
def spectral_cluster(df):
	from sklearn import cluster
	
	#Placeholder for the names of videos and their vectors of hashed values
	names = []
	vals = []

	#Split our result so we can do the clustering
	for i in df[0]:
		names.append(i[0][:-4])
		vals.append(i[1])

	print('Data ready for clustering...')

	#Here is the actual clustering, we specify number of clusters and nearest neighbor affinity
	# we also decided to run this in two jobs to try and minimize run time
	spectral = cluster.SpectralClustering(
        n_clusters=970, eigen_solver='arpack',
        affinity="nearest_neighbors", n_init=20, n_jobs=2, assign_labels = 'discretize')

	#Fit our data to the clustering method
	spectral.fit(vals)
	print('Clustering done...')

	print(print('Runtime: ' + str(time.time() - start)))
	
	#Zip our clusters and video names together so we can check the results
	zipped = zip(spectral.labels_, names)

	#Placeholder for results
	result_sets = defaultdict(set)

	for label,vid in zipped:
		#Create the results dictionary
		result_sets[label].add(vid)

	#Check the results
	check_res(list(result_sets.values()))
	

#Here we do the multiprocessing
def multipro(files):
	import multiprocessing
	#We need to set the start method of processes to 'spawn' so it works with openCV/cv2
	multiprocessing.set_start_method('spawn')

	#Placeholder for our data from the files
	df = []

	#Create a multiprocessing pool of 10 threads
	p = multiprocessing.Pool(10)

	#Map the threads to our function above with the ids as parameter, set chunksize to 970 (970*10=9700)
	df.append(p.map(file_query, files, chunksize=970))

	print('Files read and hashed...')

	return df

#Here we go through a video and hash each frame as well as do feature hashing
def file_query(filename):

	#Capture the video
	cap = cv2.VideoCapture('./videos/' + filename)
	
	#Placeholder for the hash of each frame
	hashes = []

	#Number of frames
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	for i in range(0,length):
		#We check where in the video we are as we want to weight the end down
		if i > (0.1*length) and i < (0.9*length):
			ret ,frame = cap.read()
			if ret:
				
				#Get the height and width of the frame
				height = np.size(frame, 0)
				width = np.size(frame, 1)

				#Here we crop, grayscale and equlize the frame
				x_start = int(width/2 - width*0.2)
				x_end = int(width/2 + width*0.2)
				y_start = int(height/2 - height*0.2)
				y_end = int(height/2 + height*0.2)
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				frame = frame[y_start:y_end, x_start:x_end]
				frame = cv2.equalizeHist(frame)

				#Hash the frame and save the hash
				imh = (str(imagehash.average_hash(Image.fromarray(frame), hash_size=5)))
				hashes.append(int(imh, 16))

			else:
				break
		else:
			ret ,frame = cap.read()
			if ret:

				#Get the height and width of the frame
				height = np.size(frame, 0)
				width = np.size(frame, 1)

				#Here we crop, grayscale and equlize the frame
				x_start = int(width/2 - width*0.2)
				x_end = int(width/2 + width*0.2)
				y_start = int(height/2 - height*0.2)
				y_end = int(height/2 + height*0.2)
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				frame = frame[y_start:y_end, x_start:x_end]
				frame = cv2.equalizeHist(frame)
				
				#Set the weight to half
				frame = frame*0.5

				#Hash the frame and save the hash
				imh = (str(imagehash.average_hash(Image.fromarray(frame), hash_size=5)))
				hashes.append(int(imh, 16))

			else:
				break

	#Release the capture
	cap.release()

	#Here we do feature hashing

	#Size of bucket, determined by trial-n-error
	bucket = 625
	#Vector for feature hashing
	vec = np.zeros(bucket)
	
	for i in hashes:
		i_hash = hash(i)
		vec[i_hash % bucket] += 1
	
	return (filename, vec)


if __name__ == '__main__':
	print('Starting...')

	start = time.time()

	df = multipro(os.listdir('./videos/'))

	spectral_cluster(df)
	
	print('Runtime with result checking: ' + str(time.time() - start))
