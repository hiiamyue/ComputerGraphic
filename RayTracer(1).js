 
function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Ray Tracer';
	UI.titleShort = 'RayTracerSimple';
	UI.numFrames = 100;
	UI.maxFPS = 24;
	UI.renderWidth = 1600;
	UI.renderHeight = 800;

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `RaytracingDemoFS - GL`,
		id: `RaytracingDemoFS`,
		initialValue: `#define SOLUTION_CYLINDER_AND_PLANE
#define SOLUTION_SHADOW
#define SOLUTION_REFLECTION_REFRACTION
#define SOLUTION_FRESNEL
#define SOLUTION_BOOLEAN

precision highp float;
uniform ivec2 viewport; 

struct PointLight {
	vec3 position;
	vec3 color;
};

struct Material {
	vec3  diffuse;
	vec3  specular;
	float glossiness;
	float reflection;
	float refraction;
	float ior;
};

struct Sphere {
	vec3 position;
	float radius;
	Material material;
};

struct Plane {
	vec3 normal;
	float d;
	Material material;
};

struct Cylinder {
	vec3 position;
	vec3 direction;  
	float radius;
	Material material;
};

#define BOOLEAN_MODE_AND 0			// and 
#define BOOLEAN_MODE_MINUS 1			// minus 

struct Boolean {
    Sphere spheres[2];
    int mode;
};


const int lightCount = 2;
const int sphereCount = 3;
const int planeCount = 1;
const int cylinderCount = 2;
const int booleanCount = 2; 

struct Scene {
	vec3 ambient;
	PointLight[lightCount] lights;
	Sphere[sphereCount] spheres;
	Plane[planeCount] planes;
	Cylinder[cylinderCount] cylinders;
    Boolean[booleanCount] booleans;
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
	bool enteringPrimitive;
};

HitInfo getEmptyHit() {
	return HitInfo(
		false, 
		0.0, 
		vec3(0.0), 
		vec3(0.0), 
		Material(vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0),
		false);
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

// Get the smallest t in an interval.
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

HitInfo intersectSphere(const Ray ray, const Sphere sphere, const float tMin, const float tMax) {
              
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
		
		//Checking if we're inside the sphere by checking if the ray's origin is inside. If we are, then the normal 
		//at the intersection surface points towards the center. Otherwise, if we are outside the sphere, then the normal 
		//at the intersection surface points outwards from the sphere's center. This is important for refraction.
      	vec3 normal = 
          	length(ray.origin - sphere.position) < sphere.radius + 0.001? 
          	-normalize(hitPosition - sphere.position): 
      		normalize(hitPosition - sphere.position);      
		
		//Checking if we're inside the sphere by checking if the ray's origin is inside,
		// but this time for IOR bookkeeping. 
		//If we are inside, set a flag to say we're leaving. If we are outside, set the flag to say we're entering.
		//This is also important for refraction.
		bool enteringPrimitive = 
          	length(ray.origin - sphere.position) < sphere.radius + 0.001 ? 
          	false:
		    true; 

        return HitInfo(
          	true,
          	smallestTInInterval,
          	hitPosition,
          	normal,
          	sphere.material,
			enteringPrimitive);
    }
    return getEmptyHit();
}

HitInfo intersectPlane(const Ray ray,const Plane plane, const float tMin, const float tMax) {
#ifdef SOLUTION_CYLINDER_AND_PLANE
	//Substitute ray equation into plane dot product equation to get t value that ray intersects with the plane ->
	//normal * a point on the plane = - distance.
	float a = dot(plane.normal,ray.origin);
	float b = dot(plane.normal,ray.direction);
	float t =(plane.d-a)/b; 
	// t cannot be 0.0 since it will mean the ray is perpendicular to the plane.
	if(t>0.0){
		float smallestTInInterval;
			if(!getSmallestTInInterval(t, t, tMin, tMax, smallestTInInterval)) {
			  return getEmptyHit();
			}
	   //substitute t value back to ray equation to get hit position.
		vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;   

		//does not matter here because it is hard to define whether inside or outside a plane.
		bool enteringPrimitive = true;

	   return HitInfo(
			true,
			smallestTInInterval,
			hitPosition,
			plane.normal,
			plane.material,
			enteringPrimitive);
		}
#endif  
		return getEmptyHit();
}

float lengthSquared(vec3 x) {
	return dot(x, x);
}

HitInfo intersectCylinder(const Ray ray, const Cylinder cylinder, const float tMin, const float tMax) {
#ifdef SOLUTION_CYLINDER_AND_PLANE
	//substitute ray equation into cylinder equation to get t value so that ray intersect with cylinder surface.
	//simplify substituted equation into form at^2+bt+c = 0 , calculate a,b and c.
	vec3 a1 =(ray.direction-dot(ray.direction,cylinder.direction) *cylinder.direction);
	float a = length(a1)*length(a1);
	vec3 s = ray.origin-cylinder.position;
	float b = 2.0*dot(a1,(s-dot(s,cylinder.direction)*cylinder.direction));
	float c = length(s-dot(s,cylinder.direction)*cylinder.direction)*length(s-dot(s,cylinder.direction)*cylinder.direction)-cylinder.radius*cylinder.radius;
	float D = b*b - 4.0*a*c;
	// if D<0.0,there are no intersections whereas is if D == 0.0, ray is perpendicular to the surface .Both case we do not cast ray.
	if (D >0.0){
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);
      	
      	float smallestTInInterval;
      	if(!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
          return getEmptyHit();
        }
		vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;
		
		//checking if rays origin is inside the cylinders...if inside the cylinder then normal points inwards,vice versa
		//calculate normal with given smallestTInInterval value
		vec3 normal = 
          	length(ray.origin - cylinder .position) < cylinder.radius + 0.001? 
          	-normalize((s+ray.direction*smallestTInInterval)-dot(cylinder.direction,s+ray.direction*smallestTInInterval)*cylinder.direction): 
      		normalize((s+ray.direction*smallestTInInterval)-dot(cylinder.direction,s+ray.direction*smallestTInInterval)*cylinder.direction);     
         
		//checking if rays origin is inside the cylinders
		///If we are inside, set a flag to say we're leaving. If we are outside, set the flag to say we're entering.
		bool enteringPrimitive = 
			length(ray.origin - cylinder.position) < cylinder.radius + 0.001 ? 
          	false:
		    true; 
          	
		return HitInfo(
		true,
        smallestTInInterval,
        hitPosition,
        normal,
        cylinder.material,
		enteringPrimitive);
	
		
	}
		
#endif  
    return getEmptyHit();
}

bool inside(const vec3 position, const Sphere sphere) {
	return length(position - sphere.position) < sphere.radius;
}

HitInfo intersectBoolean(const Ray ray, const Boolean boolean, const float tMin, const float tMax) {
#ifdef SOLUTION_BOOLEAN
	//HitInfo structure could have a method to flip normal to opposite direction;
	Sphere A = boolean.spheres[0];
	Sphere B = boolean.spheres[1];
	HitInfo hitInfoA = intersectSphere(ray, A, tMin,tMax);
	HitInfo hitInfoB = intersectSphere(ray, B, tMin,tMax);
	
	vec3 AinB =B.position-hitInfoA.position;
	vec3 BinA = A.position-hitInfoB.position;
	float aInb = length(AinB);
	float bIna = length(BinA);
	
	//mode AND
	if(boolean.mode==0){
		//getting points in A that is also in B by calculating if the distance of point in A and position B is less than the radius of B.vice versa.
		
		if(aInb <B.radius){
			return hitInfoA;
		}
		if(bIna <A.radius){
			return hitInfoB;
		}
	}
	// mode Minus
	if(boolean.mode ==1){
		//points on B but not A
		if(bIna>A.radius){
			return hitInfoB;
		}
		//use B's t value to replace tMin so we get A points intersecting B sphere
		hitInfoA =intersectSphere(ray,A,hitInfoB.t,tMax);
		bool a =inside(hitInfoA.position,B);
		if(a == true){
			hitInfoA.normal *=-1.0;
			//negative normal because now light casted from other side of point
			return hitInfoA;
		}
		
	}
	
#else
    // Put your code for the boolean task in the #ifdef above!
#endif
    return getEmptyHit();
}

uniform float time;

HitInfo getBetterHitInfo(const HitInfo oldHitInfo, const HitInfo newHitInfo) {
	if(newHitInfo.hit)
  		if(newHitInfo.t < oldHitInfo.t)  // No need to test for the interval, this has to be done per-primitive
          return newHitInfo;
  	return oldHitInfo;
}

HitInfo intersectScene(const Scene scene, const Ray ray, const float tMin, const float tMax) {
	HitInfo bestHitInfo;
	bestHitInfo.t = tMax;
	bestHitInfo.hit = false;


	for (int i = 0; i < booleanCount; ++i) {
    	bestHitInfo = getBetterHitInfo(bestHitInfo, intersectBoolean(ray, scene.booleans[i], tMin, tMax));
	}

		for (int i = 0; i < planeCount; ++i) {
			bestHitInfo = getBetterHitInfo(bestHitInfo, intersectPlane(ray, scene.planes[i], tMin, tMax));
		}

		for (int i = 0; i < sphereCount; ++i) {
			bestHitInfo = getBetterHitInfo(bestHitInfo, intersectSphere(ray, scene.spheres[i], tMin, tMax));
		}

		for (int i = 0; i < cylinderCount; ++i) {
			bestHitInfo = getBetterHitInfo(bestHitInfo, intersectCylinder(ray, scene.cylinders[i], tMin, tMax));
		}
	
	return bestHitInfo;
}

vec3 shadeFromLight(
  const Scene scene,
  const Ray ray,
  const HitInfo hit_info,
  const PointLight light)
{ 
  vec3 hitToLight = light.position - hit_info.position;
  
  vec3 lightDirection = normalize(hitToLight);
  vec3 viewDirection = normalize(hit_info.position - ray.origin);
  vec3 reflectedDirection = reflect(viewDirection, hit_info.normal);
  float diffuse_term = max(0.0, dot(lightDirection, hit_info.normal));
  float specular_term  = pow(max(0.0, dot(lightDirection, reflectedDirection)), hit_info.material.glossiness);

#ifdef SOLUTION_SHADOW
	// cast ray from hit point to the light
	Ray currentRay;
	currentRay.origin = hit_info.position;
	currentRay.direction = hitToLight;
	//see if the ray intersect with any objects in scene
	HitInfo intersections = intersectScene(scene, currentRay, 0.001, 10000.0);
	float visibility = 1.0;
	float d = length(lightDirection);
	
	//if theres a intersection and the object is in font of light source, set visibility to 0.0 to cast shadow
	//compare normalised distance with t to filter out intersections with objects beyond light source
	// also filter out the case when t< 0.0 meaning object are behind the hit point
	if (intersections.hit) {
    	if(intersections.t < d && intersections.t >0.0){
			visibility = 0.0;
		}
	}
#else
  // Put your shadow test here
  float visibility = 1.0;
#endif
  return 	visibility * 
    		light.color * (
    		specular_term * hit_info.material.specular +
      		diffuse_term * hit_info.material.diffuse);
}

vec3 background(const Ray ray) {
  // A simple implicit sky that can be used for the background
  return vec3(0.2) + vec3(0.8, 0.6, 0.5) * max(0.0, ray.direction.y);
}

// It seems to be a WebGL issue that the third parameter needs to be inout instea dof const on Tobias' machine
vec3 shade(const Scene scene, const Ray ray, inout HitInfo hitInfo) {
	
  	if(!hitInfo.hit) {
  		return background(ray);
  	}
  
    vec3 shading = scene.ambient * hitInfo.material.diffuse;
    for (int i = 0; i < lightCount; ++i) {
        shading += shadeFromLight(scene, ray, hitInfo, scene.lights[i]); 
    }
    return shading;
}


Ray getFragCoordRay(const vec2 frag_coord) {
  	float sensorDistance = 1.0;
  	vec2 sensorMin = vec2(-1, -0.5);
  	vec2 sensorMax = vec2(1, 0.5);
  	vec2 pixelSize = (sensorMax- sensorMin) / vec2(viewport.x, viewport.y);
  	vec3 origin = vec3(0, 0, sensorDistance);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * frag_coord, -sensorDistance));  
  
  	return Ray(origin, direction);
}

float fresnel(const vec3 viewDirection, const vec3 normal, const float sourceIOR, const float destIOR) {
#ifdef SOLUTION_FRESNEL
	//used Schlick approxmation 
	//theta is the angle between the direction from which the incident light is coming and the normal of the interface between the two media
	//https://en.wikipedia.org/wiki/Schlick%27s_approximation
	float c =dot(viewDirection,normal);
	float a = (1.0-c)*(1.0-c);
	float R0 =((sourceIOR-destIOR)/(sourceIOR+destIOR))*((sourceIOR-destIOR)/(sourceIOR+destIOR));
	float result = R0+(1.0-R0)*a;
	return result;
#else
  	// Put your code to compute the Fresnel effect in the ifdef above
	return 1.0;
#endif
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
      
    Ray initialRay = getFragCoordRay(fragCoord);  
  	HitInfo initialHitInfo = intersectScene(scene, initialRay, 0.001, 10000.0);  
  	vec3 result = shade(scene, initialRay, initialHitInfo);
	
  	Ray currentRay;
  	HitInfo currentHitInfo;
  	
  	// Compute the reflection
  	currentRay = initialRay;
  	currentHitInfo = initialHitInfo;
  	
  	// The initial strength of the reflection
  	float reflectionWeight = 1.0;
  	
  	const int maxReflectionStepCount = 2;
  	for(int i = 0; i < maxReflectionStepCount; i++) {
      
      if(!currentHitInfo.hit) break;
      
#ifdef SOLUTION_REFLECTION_REFRACTION
		Material currentMaterial = currentHitInfo.material;
		reflectionWeight *= currentMaterial.reflection;
#else
      // Put your reflection weighting code in the ifdef above
#endif
      
#ifdef SOLUTION_FRESNEL
	  reflectionWeight *=  fresnel(-currentRay.direction, currentHitInfo.normal,currentHitInfo.material.ior,currentHitInfo.material.ior);
		 
#else
      // Replace with Fresnel code in the ifdef above
      reflectionWeight *= 0.5;
#endif
      Ray nextRay;
#ifdef SOLUTION_REFLECTION_REFRACTION
		vec3 e = -currentRay.direction;
		vec3 n = currentHitInfo.normal;
		vec3 r = -e+2.0*dot(n,e)*n;
		vec3 reflectedDirection = r;
		if(dot(n,e)==0.0){
			 reflectedDirection = currentHitInfo.normal;
		}
		nextRay.direction =reflectedDirection;
		nextRay.origin = currentHitInfo.position;
		
#else
	// Put your code to compute the reflection ray in the ifdef above
#endif
      currentRay = nextRay;
      
      currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);      
            
      result += reflectionWeight * shade(scene, currentRay, currentHitInfo);
    }
  

		
  	// Compute the refraction
  	currentRay = initialRay;  
  	currentHitInfo = initialHitInfo;
   
  	// The initial medium is air
  	float currentIOR = 1.0;

  	// The initial strength of the refraction.
  	float refractionWeight = 1.0;
  
  	const int maxRefractionStepCount = 2;
  	for(int i = 0; i < maxRefractionStepCount; i++) {
      
#ifdef SOLUTION_REFLECTION_REFRACTION
		Material currentMaterial = currentHitInfo.material;
		refractionWeight *= currentMaterial.refraction;
		
#else
      // Put your refraction weighting code in the ifdef above
      reflectionWeight *= 0.5;      
#endif

#ifdef SOLUTION_FRESNEL

		refractionWeight *= 1.0- fresnel(-currentRay.direction, currentHitInfo.normal,currentIOR,currentHitInfo.material.ior);
#else
      // Put your Fresnel code in the ifdef above 
#endif      

      Ray nextRay;


#ifdef SOLUTION_REFLECTION_REFRACTION 
		bool currentEnteringPrimitive = currentHitInfo.enteringPrimitive;
		float sourceIOR = currentIOR;
		float destIOR = currentMaterial.ior;
		vec3 e = -currentRay.direction;
		vec3 n = currentHitInfo.normal;
		float angle =dot(n,e);
		
		//if the ray are outside the object,ior is air for 1.0 and destIOR will be IOR from hitInfo
		//if inside the object,sourceIOR = currentIOR and destIOR will be 1.0 for air.
		if ( currentEnteringPrimitive == false){
			sourceIOR=currentIOR;
			destIOR = 1.0;
		}
		float IOR = sourceIOR/destIOR;
		float c2 = sqrt(1.0+IOR*IOR*(angle*angle-1.0));
		vec3 refractedDirection = -IOR*e+(IOR*angle-c2)*n;
		
		if(c2< 0.0){
			break;
		}
		nextRay.direction = refractedDirection;
		nextRay.origin = currentHitInfo.position;
		currentRay = nextRay;
		currentIOR = destIOR;
#else
      float sourceIOR;
	  float destIOR;
	// Put your code to compute the reflection ray and track the IOR in the ifdef above
#endif
      currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
            
      result += refractionWeight * shade(scene, currentRay, currentHitInfo);
      
      if(!currentHitInfo.hit) break;
    }
  return result;
}

Material getDefaultMaterial() {
  return Material(vec3(0.3), vec3(0), 0.0, 0.0, 0.0, 0.0);
}

Material getPaperMaterial() {
  return Material(vec3(0.7, 0.7, 0.7), vec3(0, 0, 0), 5.0, 0.0, 0.0, 0.0);
}

Material getPlasticMaterial() {
	return Material(vec3(0.9, 0.3, 0.1), vec3(1.0), 10.0, 0.9, 0.0, 0.0);
}

Material getGlassMaterial() {
	return Material(vec3(0.0), vec3(0.0), 5.0, 1.0, 1.0, 1.5);
}

Material getSteelMirrorMaterial() {
	return Material(vec3(0.1), vec3(0.3), 20.0, 0.8, 0.0, 0.0);
}

Material getMetaMaterial() {
	return Material(vec3(0.1, 0.2, 0.5), vec3(0.3, 0.7, 0.9), 20.0, 0.8, 0.0, 0.0);
}

vec3 tonemap(const vec3 radiance) {
  const float monitorGamma = 2.0;
  return pow(radiance, vec3(1.0 / monitorGamma));
}

void main() {
	// Setup scene
	Scene scene;
	scene.ambient = vec3(0.12, 0.15, 0.2);

	scene.lights[0].position = vec3(5, 15, -5);
	scene.lights[0].color    = 0.5 * vec3(0.9, 0.5, 0.1);

	scene.lights[1].position = vec3(-15, 5, 2);
	scene.lights[1].color    = 0.5 * vec3(0.1, 0.3, 1.0);

	// Primitives
    bool soloBoolean = true;
	
	#ifdef SOLUTION_BOOLEAN
	#endif

	if(!soloBoolean) {
		scene.spheres[0].position            	= vec3(10, -5, -16);
		scene.spheres[0].radius              	= 6.0;
		scene.spheres[0].material 				= getPaperMaterial();

		scene.spheres[1].position            	= vec3(-7, -2, -13);
		scene.spheres[1].radius             	= 4.0;
		scene.spheres[1].material				= getPlasticMaterial();

		scene.spheres[2].position            	= vec3(0, 0.5, -5);
		scene.spheres[2].radius              	= 2.0;
		scene.spheres[2].material   			= getGlassMaterial();

		scene.planes[0].normal            		= normalize(vec3(0, 0.8, 0));
		scene.planes[0].d              			= -4.5;
		scene.planes[0].material				= getSteelMirrorMaterial();

		scene.cylinders[0].position            	= vec3(-1, 1, -26);
		scene.cylinders[0].direction            = normalize(vec3(-2, 2, -1));
		scene.cylinders[0].radius         		= 1.5;
		scene.cylinders[0].material				= getPaperMaterial();

		scene.cylinders[1].position            	= vec3(4, 1, -5);
		scene.cylinders[1].direction            = normalize(vec3(1, 4, 1));
		scene.cylinders[1].radius         		= 0.4;
		scene.cylinders[1].material				= getPlasticMaterial();

	} else {
		scene.booleans[0].mode = BOOLEAN_MODE_MINUS;
		
		// sphere A 
		scene.booleans[0].spheres[0].position      	= vec3(3, 0, -10);
		scene.booleans[0].spheres[0].radius      	= 2.75;
		scene.booleans[0].spheres[0].material      	= getPaperMaterial();
		
		// sphere B
		scene.booleans[0].spheres[1].position      	= vec3(6, 1, -13);	
		scene.booleans[0].spheres[1].radius      	= 4.0;
		scene.booleans[0].spheres[1].material      	= getPaperMaterial();
		
		
		scene.booleans[1].mode = BOOLEAN_MODE_AND;
		
		scene.booleans[1].spheres[0].position      	= vec3(-3.0, 1, -12);
		scene.booleans[1].spheres[0].radius      	= 4.0;
		scene.booleans[1].spheres[0].material      	= getPaperMaterial();
		
		scene.booleans[1].spheres[1].position      	= vec3(-6.0, 1, -12);	
		scene.booleans[1].spheres[1].radius      	= 4.0;
		scene.booleans[1].spheres[1].material      	= getMetaMaterial();
		

		scene.planes[0].normal            		= normalize(vec3(0, 0.8, 0));
		scene.planes[0].d              			= -4.5;
		scene.planes[0].material				= getSteelMirrorMaterial();
		
		scene.lights[0].position = vec3(-5, 25, -5);
		scene.lights[0].color    = vec3(0.9, 0.5, 0.1);

		scene.lights[1].position = vec3(-15, 5, 2);
		scene.lights[1].color    = 0.0 * 0.5 * vec3(0.1, 0.3, 1.0);
		
	}

	// Compute color for fragment
	gl_FragColor.rgb = tonemap(colorForFragment(scene, gl_FragCoord.xy));
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
		title: `RaytracingDemoVS - GL`,
		id: `RaytracingDemoVS`,
		initialValue: `attribute vec3 position;
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
  
    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	 return UI; 
}//!setup

var gl;
function initGL(canvas) {
	try {
		gl = canvas.getContext("experimental-webgl");
		gl.viewportWidth = canvas.width;
		gl.viewportHeight = canvas.height;
	} catch (e) {
	}
	if (!gl) {
		alert("Could not initialise WebGL, sorry :-(");
	}
}

function getShader(gl, id) {
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

RaytracingDemo.prototype.initShaders = function() {

	this.shaderProgram = gl.createProgram();

	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoVS"));
	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoFS"));
	gl.linkProgram(this.shaderProgram);

	if (!gl.getProgramParameter(this.shaderProgram, gl.LINK_STATUS)) {
		alert("Could not initialise shaders");
	}

	gl.useProgram(this.shaderProgram);

	this.shaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.shaderProgram, "position");
	gl.enableVertexAttribArray(this.shaderProgram.vertexPositionAttribute);

	this.shaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.shaderProgram, "projectionMatrix");
	this.shaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.shaderProgram, "modelViewMatrix");
}

RaytracingDemo.prototype.initBuffers = function() {
	this.triangleVertexPositionBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	
	var vertices = [
		 -1,  -1,  0,
		 -1,  1,  0,
		 1,  1,  0,

		 -1,  -1,  0,
		 1,  -1,  0,
		 1,  1,  0,
	 ];
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
	this.triangleVertexPositionBuffer.itemSize = 3;
	this.triangleVertexPositionBuffer.numItems = 3 * 2;
}

function getTime() {  
	var d = new Date();
	return d.getMinutes() * 60.0 + d.getSeconds() + d.getMilliseconds() / 1000.0;
}

RaytracingDemo.prototype.drawScene = function() {
			
	var perspectiveMatrix = new J3DIMatrix4();	
	perspectiveMatrix.setUniform(gl, this.shaderProgram.projectionMatrixUniform, false);

	var modelViewMatrix = new J3DIMatrix4();	
	modelViewMatrix.setUniform(gl, this.shaderProgram.modelviewMatrixUniform, false);

	gl.uniform1f(gl.getUniformLocation(this.shaderProgram, "time"), getTime());
	
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	gl.vertexAttribPointer(this.shaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
	
	gl.uniform2iv(gl.getUniformLocation(this.shaderProgram, "viewport"), [getRenderTargetWidth(), getRenderTargetHeight()]);

	gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);
}

RaytracingDemo.prototype.run = function() {
	this.initShaders();
	this.initBuffers();

	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
	gl.clear(gl.COLOR_BUFFER_BIT);

	this.drawScene();
};

function init() {	
	

	env = new RaytracingDemo();	
	env.run();

    return env;
}

function compute(canvas)
{
    env.initShaders();
    env.initBuffers();

    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT);

    env.drawScene();
}
