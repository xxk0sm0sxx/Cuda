#ifndef CAMERA_H_
#define CAMERA_H_

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

const float PI = 3.14159265359f;
const float PI_TWICE = PI * 2.0f;
const float PI_HALF = PI * 0.5f;
const float PI_QUARTER = PI * 0.25f;


/* Initial camera angle */
const float EYERADIUS0 = 50.0f;
const float EYEALPHA0 = PI*0.0f;
const float EYEBETA0 = PI*0.0f;
const float EYEALPHA1 = PI*0.0f;
const float EYEBETA1 = PI*0.0f;
const float FOVY = PI_QUARTER;

class Camera
{
public:
	/* Camera position & direction */
	glm::vec3 eye;
	glm::vec3 center;
	glm::vec3 upVec;
	float eyeRadius;
	float eyeAlpha, eyeBeta;

	Camera()
	{
		center = glm::vec3(0.0f, 0.0f, 0.0f);
		upVec = glm::vec3(0.0f, 1.0f, 0.0f);
		eyeRadius = EYERADIUS0;
		eyeAlpha = EYEALPHA0;
		eyeBeta = EYEBETA0;
	}

	void ComputeEye()
	{
		// alpha is angle about y, measured from z, moving in ccw direction
		// beta is angle about x, measure from z, moving in cw direction
		float cosB = glm::cos(eyeBeta);
		float sinB = glm::sin(eyeBeta);
		float cosA = glm::cos(eyeAlpha);
		float sinA = glm::sin(eyeAlpha);
		eye = center + glm::vec3(eyeRadius*cosB*sinA, eyeRadius*sinB, eyeRadius*cosB*cosA);

		if (eyeBeta > PI_HALF && eyeBeta < 3 * PI_HALF)
			upVec = glm::vec3(0, -1, 0);
		else
			upVec = glm::vec3(0, 1, 0);
	}

	glm::mat4 LookAtMatrix() const
	{
		return glm::lookAt(eye, center, upVec);
	}
};

#endif