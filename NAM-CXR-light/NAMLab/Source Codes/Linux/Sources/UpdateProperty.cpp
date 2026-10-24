/*************************************************
author: Bowen Yang
date:2019/10/22
describe: update adjacency list and features.
**************************************************/
#include "NAM.h"
#include "unionFind.h"


void UpdateProperty(list<Vec2i> &adjacent, list<float> &DC, list<float> &DBC, list<float> &DT, list<float> &SI, list<float> &DN, list<float> &Sum, list<float> &Count, list<float> &DAB, list<float> &DW, float& SumDC, float &SumDT, float &SumDBC, const vector<Region> &region, vector<int> &parent, const int &R1, const int &R2, const float &t, const float &ThNc)
{
	list<Vec2i>::iterator adjItr;
	list<float>::iterator sumItr, countItr, siItr, dbcItr, dcItr, dabItr, dwItr, dtItr, dnItr;

	//remove some items from the original lists, and use them to form new temporary lists. Finally, form updated lists based on these temporary lists, and merge the updated lists with the original lists. 
	list<Vec2i> adjSel, adjUP;
	list<float> SumSel, CountSel, SISel, SumUP, CountUP, SIUP;
	list<float> DBCUP, DCUP, DABUP, DWUP, DTUP, DNUP;

	adjItr = adjacent.begin();
	siItr = SI.begin();
	dbcItr = DBC.begin();
	dcItr = DC.begin();
	dtItr = DT.begin();
	dnItr = DN.begin();
	sumItr = Sum.begin();
	countItr = Count.begin();
	dabItr = DAB.begin();
	dwItr = DW.begin();
	//find items related to R1 or R2.
	while (adjItr != adjacent.end())
	{
		const Vec2i &pos = (*adjItr);
		if (pos[0] == R1 || pos[0] == R2 || pos[1] == R1 || pos[1] == R2)
		{
			//form new temporary lists.
			adjSel.push_back((*adjItr));
			SumSel.push_back((*sumItr));
			CountSel.push_back((*countItr));
			SISel.push_back((*siItr));

			SumDBC -= (*dbcItr);
			SumDC -= (*dcItr);
			SumDT -= (*dtItr);

			//remove items from the original lists.
			adjItr = adjacent.erase(adjItr);
			siItr = SI.erase(siItr);
			dbcItr = DBC.erase(dbcItr);
			dcItr = DC.erase(dcItr);
			dtItr = DT.erase(dtItr);
			dnItr = DN.erase(dnItr);
			sumItr = Sum.erase(sumItr);
			countItr = Count.erase(countItr);
			dabItr = DAB.erase(dabItr);
			dwItr = DW.erase(dwItr);
		}
		else
		{
			adjItr++;
			siItr++;
			dbcItr++;
			dcItr++;
			dtItr++;
			dnItr++;
			sumItr++;
			countItr++;
			dabItr++;
			dwItr++;
		}
	}

	
	const size_t &N = parent.size();

	for (auto &x : adjSel)
	{
		x[0] = Find_Parent(parent, x[0]);
		x[1] = Find_Parent(parent, x[1]);
	}

	/* deduplicate and update */
	map<long long, float> adjMap, sumMap, countMap, siMap;
	for (auto &x : adjSel)
	{
		if (x[0] == x[1])continue;	//remove items like (Ri, Ri)
		long long pos = (long long)x[0] * N + (long long)x[1];
		adjMap[pos] = 1;
		sumMap[pos] = 0;
		countMap[pos] = 0;
		siMap[pos] = 0;
	}

	siItr = SISel.begin();
	sumItr = SumSel.begin();
	countItr = CountSel.begin();
	for (auto &x : adjSel)
	{
		if (x[0] != x[1])
		{
			long long pos = (long long)x[0] * N + (long long)x[1];
			sumMap[pos] += (*sumItr);
			countMap[pos] += (*countItr);
			siMap[pos] += (*siItr);
		}
		siItr++;
		sumItr++;
		countItr++;
	}

	for (auto &x : adjMap)
	{
		Vec2i data(x.first / (long long)N, x.first % (long long)N);
		adjUP.push_back(data);
		long long pos1 = (long long)data[0] * N + (long long)data[1];
		long long pos2 = (long long)data[1] * N + (long long)data[0];
		SumUP.push_back(max(sumMap[pos1], sumMap[pos2]));
		CountUP.push_back(max(countMap[pos1], countMap[pos2]));
		SIUP.push_back(min(siMap[pos1], siMap[pos2]));
	}
	//DBCUP, DCUP, DABUP, DWUP, DTUP, DNUP, DXYUP
	ComputeDBC(SumUP, CountUP, DBCUP);
	ComputeDC(region, adjUP, DCUP);
	ComputeDAB(region, adjUP, DABUP);
	ComputeDW(region, adjUP, DWUP);
	ComputeDT(DABUP, DWUP, DTUP);
	ComputeDN(region, adjUP, DNUP, t, ThNc);

	for (auto &x : DBCUP)
		SumDBC += x;
	for (auto &x : DCUP)
		SumDC += x;
	for (auto &x : DTUP)
		SumDT += x;

	/* merge lists */
	adjItr = adjacent.end();
	siItr = SI.end();
	dbcItr = DBC.end();
	dcItr = DC.end();
	dtItr = DT.end();
	dnItr = DN.end();
	sumItr = Sum.end();
	countItr = Count.end();
	dabItr = DAB.end();
	dwItr = DW.end();
	adjacent.splice(adjItr, adjUP);
	SI.splice(siItr, SIUP);
	DBC.splice(dbcItr, DBCUP);
	DC.splice(dcItr, DCUP);
	DT.splice(dtItr, DTUP);
	DN.splice(dnItr, DNUP);
	Sum.splice(sumItr, SumUP);
	Count.splice(countItr, CountUP);
	DAB.splice(dabItr, DABUP);
	DW.splice(dwItr, DWUP);
}
