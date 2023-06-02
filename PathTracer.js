
function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Path Tracer';
	UI.titleShort = 'PathTracer';
	UI.numFrames = 10;
	UI.maxFPS = 1000;
	UI.renderWidth = 256;
	UI.renderHeight = 128;

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `Raytracing`,
		id: `TraceFS`,
		initialValue: `#define SOLUTION_LIGHT
#define SOLUTION_BOUNCE
#define SOLUTION_THROUGHPUT
#define SOLUTION_HALTON
#define SOLUTION_AA
#define SOLUTION_IS
//#define SOLUTION_MB

precision highp float;

#define M_PI 3.1415

struct Material {
	#ifdef SOLUTION_LIGHT
	vec3 emission;
	#endif
	vec3 diffuse;
	vec3 specular;
	float glossiness;
};

struct Sphere {
	vec3 position;
#ifdef SOLUTION_MB
	vec3 motionBlur;
#endif
	float radius;
	Material material;
};

struct Plane {
	vec3 normal;
	float d;
	Material material;
};

const int sphereCount = 4;
const int planeCount = 4;
const int emittingSphereCount = 2;
#ifdef SOLUTION_BOUNCE
const int maxPathLength = 2;
#else
const int maxPathLength = 1;
#endif

struct Scene {
	Sphere[sphereCount] spheres;
	Plane[planeCount] planes;
};

struct Ray {
	vec3 origin;
	vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
	bool hit;
	float t;
	vec3 position;
	vec3 normal;
	Material material;
};

// Contains info to sample a direction and this directions probability
struct DirectionSample {
	vec3 direction;
	float probability;
};

HitInfo getEmptyHit() {
	Material emptyMaterial;
	#ifdef SOLUTION_LIGHT
	emptyMaterial.emission = vec3(0.0);
	#endif
	emptyMaterial.diffuse = vec3(0.0);
	emptyMaterial.specular = vec3(0.0);
	emptyMaterial.glossiness = 0.0;
	return HitInfo(false, 0.0, vec3(0.0), vec3(0.0), emptyMaterial);
}

// Sorts the two t values such that t1 is smaller than t2
void sortT(inout float t1, inout float t2) {
	// Make t1 the smaller t
	if(t2 < t1)  {
		float temp = t1;
		t1 = t2;
		t2 = temp;
	}
}

// Tests if t is in an interval
bool isTInInterval(const float t, const float tMin, const float tMax) {
	return t > tMin && t < tMax;
}

// Get the smallest t in an interval
bool getSmallestTInInterval(float t0, float t1, const float tMin, const float tMax, inout float smallestTInInterval) {

	sortT(t0, t1);

	// As t0 is smaller, test this first
	if(isTInInterval(t0, tMin, tMax)) {
		smallestTInInterval = t0;
		return true;
	}

	// If t0 was not in the interval, still t1 could be
	if(isTInInterval(t1, tMin, tMax)) {
		smallestTInInterval = t1;
		return true;
	}

	// None was
	return false;
}

// Converts a random integer in 15 bits to a float in (0, 1)
float randomInetegerToRandomFloat(int i) {
	return float(i) / 32768.0;
}

// Returns a random integer for every pixel and dimension that remains the same in all iterations
int pixelIntegerSeed(const int dimensionIndex) {
	vec3 p = vec3(gl_FragCoord.xy, dimensionIndex);
	vec3 r = vec3(23.14069263277926, 2.665144142690225,7.358926345 );
	return int(32768.0 * fract(cos(dot(p,r)) * 123456.0));
}

// Returns a random float for every pixel that remains the same in all iterations
float pixelSeed(const int dimensionIndex) {
	return randomInetegerToRandomFloat(pixelIntegerSeed(dimensionIndex));
}

// The global random seed of this iteration
// It will be set to a new random value in each step
uniform int globalSeed;
int randomSeed;
void initRandomSequence() {
	randomSeed = globalSeed + pixelIntegerSeed(0);
}

// Computes integer  x modulo y not available in most WEBGL SL implementations
int mod(const int x, const int y) {
	return int(float(x) - floor(float(x) / float(y)) * float(y));
}

// Returns the next integer in a pseudo-random sequence
int rand() {
	randomSeed = randomSeed * 1103515245 + 12345;
	return mod(randomSeed / 65536, 32768);
}

// Returns the next float in this pixels pseudo-random sequence
float uniformRandom() {
	return randomInetegerToRandomFloat(rand());
}

// Returns the ith prime number for the first 20
const int maxDimensionCount = 10;
int prime(const int index) {
	if(index == 0) return 2;
	if(index == 1) return 3;
	if(index == 2) return 5;
	if(index == 3) return 7;
	if(index == 4) return 11;
	if(index == 5) return 13;
	if(index == 6) return 17;
	if(index == 7) return 19;
	if(index == 8) return 23;
	if(index == 9) return 29;
	if(index == 10) return 31;
	if(index == 11) return 37;
	if(index == 12) return 41;
	if(index == 13) return 43;
	if(index == 14) return 47;
	if(index == 15) return 53;
	return 2;
}

#ifdef SOLUTION_HALTON
#endif

float halton(const int sampleIndex, const int dimensionIndex) {
	#ifdef SOLUTION_HALTON
	float base= float(prime(dimensionIndex));
	float index = float(sampleIndex);
	
	float f =1.0;
	float r =0.0;
	
	for (int i = 0;i<10000;i++){
		if (index >0.0){
			f /=base;
			r += f*float(mod(float(index),float(base)));
			index = floor(index/base);
		}
		else{
			break;
		}
	}
	//To off set the strauctured pattern, I use a random number that is  generated per-pixel and per-dimension but nor per sample
	//The pixelSeed function provides a random float ξi,k for every pixel and dimension that remains the same in all iterations
	//Add the ξi,k to halton random sample r.
	//since both r and ξi,k range from 0 to 1, adding them up gave us range from 0-2
	// We need return value in range 0 to 1, so we use fract () to get the fractional part of sample
	r +=  pixelSeed(dimensionIndex);

	return fract(r);
	#else
	// Put your implementation of halton in the #ifdef above 
	return 0.0;
	#endif
}

// This is the index of the sample controlled by the framework.
// It increments by one in every call of this shader
uniform int baseSampleIndex;

// Returns a well-distributed number in (0,1) for the dimension dimensionIndex
float sample(const int dimensionIndex) {
	#ifdef SOLUTION_HALTON
	return halton(baseSampleIndex,dimensionIndex);
	#else
	// Use the Halton sequence for variance reduction in the #ifdef above
	return uniformRandom();
	#endif
}

// This is a helper function to sample two-dimensionaly in dimension dimensionIndex
vec2 sample2(const int dimensionIndex) {
	return vec2(sample(dimensionIndex + 0), sample(dimensionIndex + 1));
}

vec3 sample3(const int dimensionIndex) {
	return vec3(sample(dimensionIndex + 0), sample(dimensionIndex + 1), sample(dimensionIndex + 2));
}

// This is a register of all dimensions that we will want to sample.
// Thanks to Iliyan Georgiev from Solid Angle for explaining proper housekeeping of sample dimensions in ranomdized Quasi-Monte Carlo
//
// So if we want to use lens sampling, we call sample(LENS_SAMPLE_DIMENSION).
//
// There are infinitely many path sampling dimensions.
// These start at PATH_SAMPLE_DIMENSION.
// The 2D sample pair for vertex i is at PATH_SAMPLE_DIMENSION + PATH_SAMPLE_DIMENSION_MULTIPLIER * i + 0
#define ANTI_ALIAS_SAMPLE_DIMENSION 0
#define TIME_SAMPLE_DIMENSION 1
#define PATH_SAMPLE_DIMENSION 3

// This is 2 for two dimensions and 2 as we use it for two purposese: NEE and path connection
#define PATH_SAMPLE_DIMENSION_MULTIPLIER (2 * 2)

vec3 getEmission(const Material material, const vec3 normal) {
	#ifdef SOLUTION_LIGHT
	
	//Gamma:
	//Gamma is used for gamma-corrected at Tonemapping to convert physical linear units brightness produced by simulation into non-linear perceived 		//brightness.
	//The human perception of brightness has greater sensitivity to relative differences between darker tones than between lighter tones
	//If images are not gamma-correlated, they allocate too many bits or too much bandwidth to highlights that humans cannot differentiate
	//In this CW, it is expressed using power law expression where output = A * pow(input,gamma)
	//In this cw, A is luminance and gamma = encoded gamma which <1
	
	return material.emission;
	#else
	// This is wrong. It just returns the diffuse color so that you see something to be sure it is working.
	return material.diffuse;
	#endif
}

vec3 getReflectance(const Material material, const vec3 normal, const vec3 inDirection, const vec3 outDirection) {
	#ifdef SOLUTION_THROUGHPUT
	//This is a BRDF function, and it returns proportions of light reflected from incoming light to outgoing light
	//It is a important part of rendering equation and is multiplied with throughput at later stage
	//to get correct amount of reflected light
	
	//retrieve material propeties 
	vec3 d = material.diffuse;
	vec3 s = material.specular;
	float n = material.glossiness;
	
	//calculate Physically-correct Phong uses the normalization factor n+2/2π 
	// by using normalisation , the scaling of light decreases according to n value
	//kd is normalised diffuse factor
	
	vec3 kd = d/ M_PI;
	//ks id normalised specular factor
	
	vec3 Ks = s*((n+2.0)/(2.0*M_PI));
	
	// The r is the light reflection unit vector,mirror of inDirection about normal
	// calculated using the Formula 𝑟=𝑑−2(𝑑⋅𝑛)𝑛
	vec3 r = inDirection -2.0*normal*dot(inDirection,normal);
	//BNDF are non negative , so any r value less than 0 is replaced with 0
	float maxR = max(0.0,dot(normalize(outDirection),normalize(r)));
	float c = pow(maxR,n);
	vec3 reflectance = kd+Ks*c;
	
	return reflectance;
	#else
	return vec3(1.0);
	#endif
}

vec3 getGeometricTerm(const Material material, const vec3 normal, const vec3 inDirection, const vec3 outDirection) {
	#ifdef SOLUTION_THROUGHPUT
	//the geometric term is the cosine angle between direction of outgoing ray and the normal 
	//In rendering equation, a geometric term is the weakening factor of outward irradiance due to incident angle, 
	//as the light flux is smeared across a surface whose area is larger than the projected area perpendicular to the ray. 
	//To get the cosine angle, we use dot product 
	//It is multiplied with throughput at later stage
	vec3 out_D = normalize(outDirection);
	float geometric = dot(normal,out_D);
	geometric = max(0.0,geometric);
	return vec3(geometric);
	#else
	return vec3(1.0);
	#endif
}

vec3 sphericalToEuclidean(float theta, float phi) {
	float x = sin(theta) * cos(phi);
	float y = sin(theta) * sin(phi);
	float z = cos(theta);
	return vec3(x, y, z);	
}
float getTheta(const float randomFloat){
		float theta =acos(2.0*randomFloat-1.0);
		return theta;
	}
float getPhi(const float randomFloat){
		float phi = randomFloat*2.0* M_PI ;
		return phi;
	}
float lengthSquared(const vec3 x) {
	return dot(x, x);
}

vec3 getRandomDirection(const int dimensionIndex) {
	#ifdef SOLUTION_BOUNCE
	float w1=sample2(dimensionIndex)[0];
	float w2 = sample2(dimensionIndex)[1];
	
	float theta = getTheta(w1);
	float phi = getPhi(w2);
	
	vec3 RandomDirection = sphericalToEuclidean(theta,phi);
	//Unit test
	//To test whether the generated random direction vector is a unit vector on a unit sphere
	//And return black vec3(0.0)if it is not a unit vector.
	// This can be done by calculating the length of the direction vector using lengthSquard() function
	// which returns dot product of direction vector and it self. I had to move the function to before getRandomDirection() so opengl recognises it.
	//Then compare where calculated length equal 1.0
	// To run the test, set bool unit_test to true.
	bool unit_test = false;
	
	if (lengthSquared(RandomDirection)!=1.0&&unit_test == true) {
		RandomDirection = vec3(0.0);
	}
	return RandomDirection;
	
	#else
	// Put your code to compute a random direction in 3D in the #ifdef above
	return vec3(0);
	#endif
}


HitInfo intersectSphere(const Ray ray, Sphere sphere, const float tMin, const float tMax) {

#ifdef SOLUTION_MB
	// while there is a motion , 
	//multiply  spheres's motion blur position and a random number  and add to original position 
	//to get a random position between original position and final position
	if (sphere.motionBlur != vec3(0.0)){
		sphere.position +=  sample(TIME_SAMPLE_DIMENSION) * sphere.motionBlur;

	}
	
	
	
#endif
	
	vec3 to_sphere = ray.origin - sphere.position;

	float a = dot(ray.direction, ray.direction);
	float b = 2.0 * dot(ray.direction, to_sphere);
	float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
	float D = b * b - 4.0 * a * c;
	if (D > 0.0)
	{
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);

		float smallestTInInterval;
		if(!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
			return getEmptyHit();
		}

		vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;

		vec3 normal =
			length(ray.origin - sphere.position) < sphere.radius + 0.001?
			-normalize(hitPosition - sphere.position) :
		normalize(hitPosition - sphere.position);

		return HitInfo(
			true,
			smallestTInInterval,
			hitPosition,
			normal,
			sphere.material);
	}
	return getEmptyHit();
}

HitInfo intersectPlane(Ray ray, Plane plane) {
	float t = -(dot(ray.origin, plane.normal) + plane.d) / dot(ray.direction, plane.normal);
	vec3 hitPosition = ray.origin + t * ray.direction;
	return HitInfo(
		true,
		t,
		hitPosition,
		normalize(plane.normal),
		plane.material);
	return getEmptyHit();
}


HitInfo intersectScene(Scene scene, Ray ray, const float tMin, const float tMax)
{
	HitInfo best_hit_info;
	best_hit_info.t = tMax;
	best_hit_info.hit = false;

	for (int i = 0; i < sphereCount; ++i) {
		Sphere sphere = scene.spheres[i];
		HitInfo hit_info = intersectSphere(ray, sphere, tMin, tMax);

		if(	hit_info.hit &&
		   hit_info.t < best_hit_info.t &&
		   hit_info.t > tMin)
		{
			best_hit_info = hit_info;
		}
	}

	for (int i = 0; i < planeCount; ++i) {
		Plane plane = scene.planes[i];
		HitInfo hit_info = intersectPlane(ray, plane);

		if(	hit_info.hit &&
		   hit_info.t < best_hit_info.t &&
		   hit_info.t > tMin)
		{
			best_hit_info = hit_info;
		}
	}

	return best_hit_info;
}

mat3 transpose(mat3 m) {
	return mat3(
		m[0][0], m[1][0], m[2][0],
		m[0][1], m[1][1], m[2][1],
		m[0][2], m[1][2], m[2][2]
	);
}

// This function creates a matrix to transform from global space into a local space oriented around the normal.
// Might be useful for importance sampling BRDF / the geometric term.
mat3 makeLocalFrame(const vec3 normal) {
	#ifdef SOLUTION_IS
	//find axis thats not parallel to normal
	vec3 majorAxis;
	if (abs(normal.x) < 0.57735026919 ) {
        majorAxis = vec3(1, 0, 0);
    } else if (abs(normal.y) < 0.57735026919 ) {
        majorAxis = vec3(0, 1, 0);
    } else {
        majorAxis = vec3(0, 0, 1);
    }
	vec3 u = normalize(cross(normal, majorAxis));
    vec3 v = cross(normal, u);
	return mat3(u,v,normal);
		
	#else
	return mat3(1.0);
	#endif
}

DirectionSample sampleDirection(const vec3 normal, const int dimensionIndex) {
	DirectionSample result;

	#ifdef SOLUTION_IS
	//One basic job Importance sampling should do is make rendering faster ,this is because Importance sampling is instead of chosing a random direction 		// after on bounce,
	// we use direction where it will most likey hit light source , so that the sample are more likely to contribute to the throughput
	// when using geometric term, we know that we know that any rays at the horizon will be heavily attenuated (specifically, cos(𝑥) ).
	//So, rays generated near the horizon will not contribute very much to the final value.
	
	//There will be no different between impotance sampled result and the one not have it
	//if there are infinitely many samples, this is beacuse if infinate many sample are given ,
	//There will be eventually enough light hitting light source that contribute to rendering equation even without IS.
	//But the one with IS will reach final state faster than the one without it.
	
	
	mat3 frame = makeLocalFrame(normal);
	float w1=sample2(dimensionIndex)[0];
	float w2 = sample2(dimensionIndex)[1];
	
	float costheta =asin(sqrt(w1)); 
	float phi = getPhi(w2);
	vec3 newDirection = sphericalToEuclidean(costheta,phi);
	
	result.direction = frame* newDirection;
	//new geometric term
	float newCosTheta = dot(result.direction,normal);
	
	result.probability = max(0.0,1.0/newCosTheta *M_PI); 
	return result;
	
	#else
	// Put yout code to compute Importance Sampling in the #ifdef above 
	result.direction = getRandomDirection(dimensionIndex);	
	result.probability = 1.0;
	#endif
	return result;
}

vec3 samplePath(const Scene scene, const Ray initialRay) {

	// Initial result is black
	vec3 result = vec3(0);

	Ray incomingRay = initialRay;
	vec3 throughput = vec3(1.0);
	for(int i = 0; i < maxPathLength; i++) {
		HitInfo hitInfo = intersectScene(scene, incomingRay, 0.001, 10000.0);

		if(!hitInfo.hit) return result;

		result += throughput * getEmission(hitInfo.material, hitInfo.normal);

		Ray outgoingRay;
		DirectionSample directionSample;
		#ifdef SOLUTION_BOUNCE
		directionSample =sampleDirection(hitInfo.normal,PATH_SAMPLE_DIMENSION+2*i);
		outgoingRay.origin = hitInfo.position;
		outgoingRay.direction =directionSample.direction;
		
		#else
		// Put your code to compute the next ray in the #ifdef above
		#endif

		#ifdef SOLUTION_THROUGHPUT
		//calculate geometric term and BRDF and multiply both of them to throughput
		// so that it become physical based with right amount of reflecting light and geometric term
		vec3 geo = getGeometricTerm(hitInfo.material,hitInfo.normal, incomingRay.direction, outgoingRay.direction);
		vec3 reflectance = getReflectance(hitInfo.material,hitInfo.normal, incomingRay.direction, outgoingRay.direction);
		throughput *= geo*reflectance;
									 
		#else
		// Compute the proper throughput in the #ifdef above 
		throughput *= 0.1;
		#endif

		#ifdef SOLUTION_IS
		throughput/=directionSample.probability/3.0;
		#else
		// Without Importance Sampling, there is nothing to do here. 
		// Put your Importance Sampling code in the #ifdef above
		#endif

		#ifdef SOLUTION_BOUNCE
		incomingRay = outgoingRay;
		#else
		// Put some handling of the next and the current ray in the #ifdef above
		#endif
	}
	return result;
}

uniform ivec2 resolution;
Ray getFragCoordRay(const vec2 fragCoord) {

	float sensorDistance = 1.0;
	vec3 origin = vec3(0, 0, sensorDistance);
	vec2 sensorMin = vec2(-1, -0.5);
	vec2 sensorMax = vec2(1, 0.5);
	vec2 pixelSize = (sensorMax - sensorMin) / vec2(resolution);
	vec3 direction = normalize(vec3(sensorMin + pixelSize * fragCoord, -sensorDistance));

	float apertureSize = 0.0;
	float focalPlane = 100.0;
	vec3 sensorPosition = origin + focalPlane * direction;
	origin.xy += -vec2(0.5);
	direction = normalize(sensorPosition - origin);

	return Ray(origin, direction);
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
	initRandomSequence();
	#ifdef SOLUTION_AA
	//To implement box blur , we randomly shift FragCoord to another point within its pixel domain
	//The Pixel domain is represented by  fragCoord.x +-0.5 and fragCoord.y+-0.5 since each pixel is 1 apart from each other
	
	//sample(ANTI_ALIAS_SAMPLE_DIMENSION) provides a random float  from 0-1 in AA dimension
	//we want random float in range -0.5 to 0.5 to fit in pixel domain so we minus 0.5.
	float rand = sample(ANTI_ALIAS_SAMPLE_DIMENSION)-0.5;
	
	//apply to shift fragCoord to another random point within pixel domain 
	vec2 sampleCoord = fragCoord+vec2(rand,rand);
		
	#else  	
	// Put your anti-aliasing code in the #ifdef above
	vec2 sampleCoord = fragCoord;
	#endif
	return samplePath(scene, getFragCoordRay(sampleCoord));
}


void loadScene1(inout Scene scene) {

	scene.spheres[0].position = vec3( 7, -2, -12);
	scene.spheres[0].radius = 2.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
	scene.spheres[0].material.emission = 150.0*vec3(0.9, 0.9, 0.5);
#endif
	scene.spheres[0].material.diffuse = vec3(0.0);
	scene.spheres[0].material.specular = vec3(0.0);
	scene.spheres[0].material.glossiness = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
	scene.spheres[0].motionBlur = vec3(0.0);
#endif
	
	scene.spheres[1].position = vec3(-8, 4, -13);
	scene.spheres[1].radius = 1.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT
	scene.spheres[1].material.emission = 150.0*vec3(0.8, 0.3, 0.1);
#endif
	scene.spheres[1].material.diffuse = vec3(0.0);
	scene.spheres[1].material.specular = vec3(0.0);
	scene.spheres[1].material.glossiness = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
	scene.spheres[1].motionBlur = vec3(0.0);
#endif
	
	scene.spheres[2].position = vec3(-2, -2, -12);
	scene.spheres[2].radius = 3.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT 
	scene.spheres[2].material.emission = vec3(0.0);
#endif  
	scene.spheres[2].material.diffuse = vec3(0.2, 0.5, 0.8);
	scene.spheres[2].material.specular = vec3(0.8);
	scene.spheres[2].material.glossiness = 40.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
	scene.spheres[2].motionBlur = vec3(-3.0,0.0,3.0);
#endif
	
	scene.spheres[3].position = vec3(3, -3.5, -14);
	scene.spheres[3].radius = 1.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT
	scene.spheres[3].material.emission = vec3(0.0);
#endif  
	scene.spheres[3].material.diffuse = vec3(0.9, 0.8, 0.8);
	scene.spheres[3].material.specular = vec3(1.0);
	scene.spheres[3].material.glossiness = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
	scene.spheres[3].motionBlur = vec3(2.0,4.0,1.0);
#endif
	
	scene.planes[0].normal = vec3(0, 1, 0);
	scene.planes[0].d = 4.5;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT 
	scene.planes[0].material.emission = vec3(0.0);
#endif
	scene.planes[0].material.diffuse = vec3(0.8);
	scene.planes[0].material.specular = vec3(0);
	scene.planes[0].material.glossiness = 50.0;    

	scene.planes[1].normal = vec3(0, 0, 1);
	scene.planes[1].d = 18.5;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT 
	scene.planes[1].material.emission = vec3(0.0);
#endif
	scene.planes[1].material.diffuse = vec3(0.9, 0.6, 0.3);
	scene.planes[1].material.specular = vec3(0.02);
	scene.planes[1].material.glossiness = 3000.0;

	scene.planes[2].normal = vec3(1, 0,0);
	scene.planes[2].d = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT
	scene.planes[2].material.emission = vec3(0.0);
#endif
	
	scene.planes[2].material.diffuse = vec3(0.2);
	scene.planes[2].material.specular = vec3(0.1);
	scene.planes[2].material.glossiness = 100.0; 

	scene.planes[3].normal = vec3(-1, 0,0);
	scene.planes[3].d = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
	scene.planes[3].material.emission = vec3(0.0);
#endif
	
	scene.planes[3].material.diffuse = vec3(0.2);
	scene.planes[3].material.specular = vec3(0.1);
	scene.planes[3].material.glossiness = 100.0; 
}


void main() {
	// Setup scene
	Scene scene;
	loadScene1(scene);

	// compute color for fragment
	gl_FragColor.rgb = colorForFragment(scene, gl_FragCoord.xy);
	gl_FragColor.a = 1.0;
}
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `Tonemapping`,
		id: `CopyFS`,
		initialValue: `precision highp float;

uniform sampler2D radianceTexture;
uniform int sampleCount;
uniform ivec2 resolution;

vec3 tonemap(vec3 color, float maxLuminance, float gamma) {
	float luminance = length(color);
	//float scale =  luminance /  maxLuminance;
	float scale =  luminance / (maxLuminance * luminance + 0.0000001);
  	return max(vec3(0.0), pow(scale * color, vec3(1.0 / gamma)));
}

void main(void) {
  vec3 texel = texture2D(radianceTexture, gl_FragCoord.xy / vec2(resolution)).rgb;
  vec3 radiance = texel / float(sampleCount);
  gl_FragColor.rgb = tonemap(radiance, 1.0, 1.6);
  gl_FragColor.a = 1.0;
}
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: ``,
		id: `VS`,
		initialValue: `
	attribute vec3 position;
	void main(void) {
		gl_Position = vec4(position, 1.0);
	}
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	 return UI; 
}//!setup


function getShader(gl, id) {

		gl.getExtension('OES_texture_float');
		//alert(gl.getSupportedExtensions());

	var shaderScript = document.getElementById(id);
	if (!shaderScript) {
		return null;
	}

	var str = "";
	var k = shaderScript.firstChild;
	while (k) {
		if (k.nodeType == 3) {
			str += k.textContent;
		}
		k = k.nextSibling;
	}

	var shader;
	if (shaderScript.type == "x-shader/x-fragment") {
		shader = gl.createShader(gl.FRAGMENT_SHADER);
	} else if (shaderScript.type == "x-shader/x-vertex") {
		shader = gl.createShader(gl.VERTEX_SHADER);
	} else {
		return null;
	}

    console.log(str);
	gl.shaderSource(shader, str);
	gl.compileShader(shader);

	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		alert(gl.getShaderInfoLog(shader));
		return null;
	}

	return shader;
}

function RaytracingDemo() {
}

function initShaders() {

	traceProgram = gl.createProgram();
	gl.attachShader(traceProgram, getShader(gl, "VS"));
	gl.attachShader(traceProgram, getShader(gl, "TraceFS"));
	gl.linkProgram(traceProgram);
	gl.useProgram(traceProgram);
	traceProgram.vertexPositionAttribute = gl.getAttribLocation(traceProgram, "position");
	gl.enableVertexAttribArray(traceProgram.vertexPositionAttribute);

	copyProgram = gl.createProgram();
	gl.attachShader(copyProgram, getShader(gl, "VS"));
	gl.attachShader(copyProgram, getShader(gl, "CopyFS"));
	gl.linkProgram(copyProgram);
	gl.useProgram(copyProgram);
	traceProgram.vertexPositionAttribute = gl.getAttribLocation(copyProgram, "position");
	gl.enableVertexAttribArray(copyProgram.vertexPositionAttribute);

}

function initBuffers() {
	triangleVertexPositionBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);

	var vertices = [
		 -1,  -1,  0,
		 -1,  1,  0,
		 1,  1,  0,

		 -1,  -1,  0,
		 1,  -1,  0,
		 1,  1,  0,
	 ];
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
	triangleVertexPositionBuffer.itemSize = 3;
	triangleVertexPositionBuffer.numItems = 3 * 2;
}


function tick() {

// 1st pass: Trace
	gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer);

	gl.useProgram(traceProgram);
  	gl.uniform1i(gl.getUniformLocation(traceProgram, "globalSeed"), Math.random() * 32768.0);
	gl.uniform1i(gl.getUniformLocation(traceProgram, "baseSampleIndex"), getCurrentFrame());
	gl.uniform2i(
		gl.getUniformLocation(traceProgram, "resolution"),
		getRenderTargetWidth(),
		getRenderTargetHeight());

	gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);
	gl.vertexAttribPointer(
		traceProgram.vertexPositionAttribute,
		triangleVertexPositionBuffer.itemSize,
		gl.FLOAT,
		false,
		0,
		0);

    	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);

	gl.disable(gl.DEPTH_TEST);
	gl.enable(gl.BLEND);
	gl.blendFunc(gl.ONE, gl.ONE);

	gl.drawArrays(gl.TRIANGLES, 0, triangleVertexPositionBuffer.numItems);

// 2nd pass: Average
   	gl.bindFramebuffer(gl.FRAMEBUFFER, null);

	gl.useProgram(copyProgram);
	gl.uniform1i(gl.getUniformLocation(copyProgram, "sampleCount"), getCurrentFrame() + 1);

	gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);
	gl.vertexAttribPointer(
		copyProgram.vertexPositionAttribute,
		triangleVertexPositionBuffer.itemSize,
		gl.FLOAT,
		false,
		0,
		0);

    	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);

	gl.disable(gl.DEPTH_TEST);
	gl.disable(gl.BLEND);

	gl.activeTexture(gl.TEXTURE0);
    	gl.bindTexture(gl.TEXTURE_2D, rttTexture);
	gl.uniform1i(gl.getUniformLocation(copyProgram, "radianceTexture"), 0);
	gl.uniform2i(
		gl.getUniformLocation(copyProgram, "resolution"),
		getRenderTargetWidth(),
		getRenderTargetHeight());

	gl.drawArrays(gl.TRIANGLES, 0, triangleVertexPositionBuffer.numItems);

	gl.bindTexture(gl.TEXTURE_2D, null);
}

function init() {
	initShaders();
	initBuffers();
	gl.clear(gl.COLOR_BUFFER_BIT);

	rttFramebuffer = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer);

	rttTexture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, rttTexture);
    	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, getRenderTargetWidth(), getRenderTargetHeight(), 0, gl.RGBA, gl.FLOAT, null);

	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, rttTexture, 0);
}

var oldWidth = 0;
var oldTraceProgram;
var oldCopyProgram;
function compute(canvas) {

	if(	getRenderTargetWidth() != oldWidth ||
		oldTraceProgram != document.getElementById("TraceFS") ||
		oldCopyProgram !=  document.getElementById("CopyFS"))
	{
		init();

		oldWidth = getRenderTargetWidth();
		oldTraceProgram = document.getElementById("TraceFS");
		oldCopyProgram = document.getElementById("CopyFS");
	}

	tick();
}
