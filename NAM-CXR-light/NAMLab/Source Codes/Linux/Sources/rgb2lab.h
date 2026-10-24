/*************************************************
author: Bowen Yang
date:2019/10/22
describe: rgb2lab
**************************************************/
#pragma once
#include<iostream>

static inline float gamma(float x)
{
	return x > 0.04045 ? pow((x + 0.055f) / 1.055f, 2.4f) : x / 12.92;
};

static void RGBToLabD65(unsigned char R, unsigned char G, unsigned char B, float &L, float &a, float &b)
{
	float tempR = gamma(R / 255.0f);
	float tempG = gamma(G / 255.0f);
	float tempB = gamma(B / 255.0f);
	float X = 0.412453 *tempR + 0.357580 *tempG + 0.180423*tempB;
	float Y = 0.212671 *tempR + 0.715160*tempG + 0.072169*tempB;
	float Z = 0.019334*tempR + 0.119193*tempG + 0.950227*tempB;
	X /= 0.950456;
	Y /= 1.0;
	Z /= 1.088754;
	float FX = X > 0.008856f ? pow(X, 1.0f / 3.0f) : (7.787f * X + 0.137931f);
	float FY = Y > 0.008856f ? pow(Y, 1.0f / 3.0f) : (7.787f * Y + 0.137931f);
	float FZ = Z > 0.008856f ? pow(Z, 1.0f / 3.0f) : (7.787f * Z + 0.137931f);
	L = Y > 0.008856f ? (116.0f * FY - 16.0f) : (903.3f * Y);
	a = 500.f *(FX - FY);
	b = 200.f *(FY - FZ);
}

static void RGBToLabD50(unsigned char R, unsigned char G, unsigned char B, float &L, float &a, float &b)
{
    //rgb->xyz
    float tempR = gamma(R / 255.0f);
    float tempG = gamma(G / 255.0f);
    float tempB = gamma(B / 255.0f);
    float X = 0.4361 *tempR + 0.3851 *tempG + 0.1431*tempB;
    float Y = 0.2225045 *tempR + 0.7168786 *tempG + 0.0606169*tempB;
    float Z = 0.0139322 *tempR + 0.0971045*tempG + 0.7141733 *tempB;

    //xyz->lab
    X /= 0.96419;
    //Y /= 1.0;
    Z /= 0.82521;
    float FX = X > 0.008856f ? pow(X, 1.0f / 3.0f) : (7.787f * X + 0.137931f);
    float FY = Y > 0.008856f ? pow(Y, 1.0f / 3.0f) : (7.787f * Y + 0.137931f);
    float FZ = Z > 0.008856f ? pow(Z, 1.0f / 3.0f) : (7.787f * Z + 0.137931f);
    L = Y > 0.008856f ? (116.0f * FY - 16.0f) : (903.3f * Y);
    a = 500.f *(FX - FY);
    b = 200.f *(FY - FZ);
}
