/*************************************************
author: Bowen Yang
date:2019/04/19
describe: The implementation of union-find algorithm.
**************************************************/
#pragma once

#include <vector>
static int Find_Parent(std::vector<int> &T, int x)
{
	int r = T[x];
	int i = x;
	while (T[r] != r)
	{
		T[i] = T[r];
		i = r;
		r = T[r];
	}
	return r;
}

static void Join(std::vector<int> &T, int x, int y)
{
	T[y] = x;
}
