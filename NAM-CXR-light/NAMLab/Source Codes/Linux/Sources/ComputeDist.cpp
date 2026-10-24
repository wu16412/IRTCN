/*************************************************
author: Bowen Yang
date:2019/04/19
describe: compute differences between regions based on formula and previous computed features.
**************************************************/
#include "NAM.h"

void ComputeDist(unordered_map<string, double> &param,const list<float> &DC, const list<float> &DBC, const list<float> &DT, const list<float> &SI, const list<float> &DN, const float &SumDC, const float &SumDBC, const float &SumDT, const float &alpha, const float &beta, const float &gamma, list<float> &Dist)
{
    list<float>::const_iterator siItr, dbcItr, dcItr, dtItr, dnItr, dxyItr;
    siItr = SI.begin();
    dbcItr = DBC.begin();
    dcItr = DC.begin();
    dtItr = DT.begin();
    dnItr = DN.begin();
    while (dbcItr != DBC.end())
    {
        float d = 0;
        d += alpha * (*dcItr++);
        d += beta * (SumDC / SumDT) * (*dtItr++);
        d += gamma * (SumDC / SumDBC) * (*dbcItr++);
        d *= (*dnItr++);
        d /= sqrt((*siItr++) + param["si"]);
        Dist.push_back(d);
    }
}

void ComputeDist(unordered_map<string, double> &param,list<float>::iterator &dcItr, list<float>::iterator &dbcItr, list<float>::iterator &dtItr, list<float>::iterator &siItr, list<float>::iterator &dnItr, const float &SumDC, const float &SumDBC, const float &SumDT, const float &alpha, const float &beta,const float &gamma, float &d)
{
    d = 0;
    d += alpha * (*dcItr);
    d += beta * (SumDC / SumDT) * (*dtItr);
    d += gamma * (SumDC / SumDBC) * (*dbcItr);
    d *= (*dnItr);
    d /= sqrt((*siItr) + param["si"]);
}
