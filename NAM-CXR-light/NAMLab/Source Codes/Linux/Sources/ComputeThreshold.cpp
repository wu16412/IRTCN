/*************************************************
author: Bowen Yang
date:2019/04/19
describe: compute Threshold for merging regions.
**************************************************/
#include "NAM.h"

void ComputeThreshold(const list<float> &Dist, const float &Percent,float &THRESHOLD)
{
	//sort Dist in an ascending order
	vector<float> RhoN;
	for (auto &x : Dist)
		RhoN.push_back((float)(x + 1e-5f));
	sort(RhoN.begin(), RhoN.end(), greater<float>{});

	//compute Threshold RhoN[(1 - Percent)/100 * RhoN.size() - 1]
	int Leng = (int)floor((1 - Percent / (float)100) *Dist.size()) - 1;
	if (Leng >= RhoN.size())
		Leng = (int)RhoN.size() - 1;
	if (Leng < 0)
		Leng = 0;
	THRESHOLD = RhoN[Leng];
}
