/*************************************************
author: Bowen Yang
date:2019/05/08
describe: Get a list of picture names in a folder.
**************************************************/
#pragma once
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <dirent.h>

void getImageList(const char* path, std::vector<std::string>& files){

    const string path0 = path;
    DIR* pDir;
    struct dirent* ptr;

    struct stat s;
    lstat(path, &s);

    if(!S_ISDIR(s.st_mode)){
        cout << "not a valid directory: " << path << endl;
        return;
    }

    if(!(pDir = opendir(path))){
        cout << "opendir error: " << path << endl;
        return;
    }
    int i = 0;
    string subFile;
    while((ptr = readdir(pDir)) != 0){
        subFile = ptr -> d_name;
        if(subFile == "." || subFile == "..")
            continue;
        files.push_back(subFile);
    }
    closedir(pDir);
}