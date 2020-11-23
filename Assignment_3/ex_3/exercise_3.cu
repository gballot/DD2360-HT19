#include <stdio.h>			// For use of the printf function
#include <sys/time.h>		// For use of gettimeofday function

#define NUM_TIMESTEPS 10000
#define ABS(a) ((a) < 0 ? -(a) : (a))
#define DT 1
# define BLOCK_SIZE 256	// Threads PER block

int BATCH_SIZE = 1000;		// Number of particles to operate on at a time
int NUM_STREAMS = 1;		// Number of CUDA Streams to use
int NUM_PARTICLES = 10000;	// # of particles to simulate, equivalent to # of threads

// Gravity field
float3 field = (float3) {0.f, 0.f, 9.8f};

// Structure for the particles
typedef struct {
  float3 position;
  float3 velocity;
} Particle;

/**
 * Can use multiple qualifiers to specify where a function will run in order
 * to reuse code that needs to be run on both host and device.
 * Change the position of the given particle based on its velocity using the
 * formula `new_position.coord = old_position.coord + velocity.coord` where
 * coord is x, y and z.
 *
 * @param particle	Particle for which a position update will be performed
 */
__host__ __device__ void updatePosition(Particle *particle) {
  particle->position.x = particle->position.x + particle->velocity.x * DT;
  particle->position.y = particle->position.y + particle->velocity.y * DT;
  particle->position.z = particle->position.z + particle->velocity.z * DT;
}

/**
 * Update the velocity of the given particle according to a field that specifies
 * the rate of change for each dimension of the particle's velocity
 *
 * @param particle	Particle for which a velocity update will be performed
 * @param field		Rate of change for each dimension (x, y, z) of a velocity
 */
__host__ __device__ void updateVelocity(Particle *particle, float3 field) {
  particle->velocity.x = particle->velocity.x + field.x * DT;
  particle->velocity.y = particle->velocity.y + field.y * DT;
  particle->velocity.z = particle->velocity.z + field.z * DT;

}

/**
 * Device implementation for the simulation of moving particles
 *
 * @param particles		List of particles for which to simulate movement
 * @param field			Values specifying the rate of change for a particle's
 *						velocity in each dimension
 * @param num_particles	# of particles, used to determine how many threads to
 *						give work if too many threads are initiated
 */
__global__ void simulateParticlesKernel(Particle *particles, float3 field,
    int num_particles) {

	// Unique ID of the current thread to determine what work to compute
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	// This thread has no work to do, exit
	if (threadId > num_particles) return;

  // Get the right particle
  Particle *particle = particles + threadId;

  // Update velocity first
  updateVelocity(particle, field);

  // Update position
  updatePosition(particle);
}

/**
 * Fill the given array with n random floats.
 *
 * @param array	Array to populate with floats.
 * @param n		Number of floats to populate the array with.
 */
void populateParticleArray(Particle *particles, int n) {
  Particle particle;

	for (int index = 0; index < n; index++) {
		// Generate random particles
    particle.position.x = 10.0 * ((float) rand() / (float) RAND_MAX);
    particle.position.y = 10.0 * ((float) rand() / (float) RAND_MAX);
    particle.position.z = 10.0 * ((float) rand() / (float) RAND_MAX);
    particle.velocity.x = 1.0 * ((float) rand() / (float) RAND_MAX);
    particle.velocity.y = 1.0 * ((float) rand() / (float) RAND_MAX);
    particle.velocity.z = 1.0 * ((float) rand() / (float) RAND_MAX);

		particles[index] = particle;
	}
}

// Entry point into the program, run each implementation of simulation and compare
// the results
int main(int argc, char **argv) {

	if (argc != 4) {
    	printf("Usage: %s <# particles> <# streams> <batch size>\n", argv[0]);
    	exit(-1);
	} else {
    	NUM_PARTICLES = atoi(argv[1]);
		NUM_STREAMS = atoi(argv[2]);
		BATCH_SIZE = atoi(argv[3]);
	}

	int numBatches = NUM_PARTICLES / BATCH_SIZE;

	// Divide particles into batches
	Particle **hostParticles = (Particle **) malloc(numBatches * sizeof(Particle *));
	Particle **devParticles = (Particle **) malloc(numBatches* sizeof(Particle *));
	for (int batch = 0; batch < numBatches; batch++) {
		// Allocate pinned (locked) memory on the host
		cudaMallocHost(&hostParticles[batch], NUM_PARTICLES * sizeof(Particle));

		// Allocate memory on the device
		cudaMalloc(&devParticles[batch], NUM_PARTICLES * sizeof(Particle));
	}

	// Fill hostParticles arrays with random floats
	populateParticleArray(hostParticles, NUM_PARTICLES);

	// Create streams
	cudaStream_t *streams = (cudaStream_t *) malloc(NUM_STREAMS * sizeof(cudaStream_t));

	// After each timestep, copy particle results back to the CPU

	int batchIndex = 0;
	for (int streamIndex = 0; streamIndex < NUM_STREAMS; streamIndex++) {
		cudaStreamCreate(&streams[streamIndex]);

		for (int timestep = 0; timestep < NUM_TIMESTEPS; timestep++) {
			for (int batch = 0; batch < numBatches; batch += NUM_STREAMS) {
				batchIndex = batch + streamIndex;

				// Copy hostParticles onto the GPU
				cudaMemcpyAsync(devParticles[batchIndex], hostParticles[batchIndex],
					NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice,
					streams[streamIndex]);

				// Round-up to the nearest multiple of BLOCK_SIZE that can hold at
				// least NUM_PARTICLES threads
				simulateParticlesKernel <<<
					(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE,
					BLOCK_SIZE, 0, streams[streamIndex]>>> (
					devParticles[batchIndex], field, NUM_PARTICLES);
				
				// Copy the result of the simulation on the device back to
				// the host into hostParticles
				cudaMemcpyAsync(hostParticles[batchIndex], devParticles[batchIndex],
					NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost,
					streams[streamIndex]);
			}
		}
	}

	// Synchronize and free the streams
	for (int streamIndex = 0; streamIndex < NUM_STREAMS; streamIndex++) {
		cudaStreamSynchonize(streams[streamIndex]);
		cudaStreamDestroy(streams[streamIndex]);
	}

	// Free the allocated memory!!!
	for (int batch = 0; batch < numBatches; batch++) {
		cudaFreeHost(hostParticles[batch]);
		cudaFree(devParticles[batch]);
	}

	free(hostParticles);
	free(devParticles);
	return 0;
}
