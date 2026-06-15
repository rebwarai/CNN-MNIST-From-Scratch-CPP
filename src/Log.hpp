// ===========================================================================
// Log.hpp
// author : @rebwar_ai
// ===========================================================================

#ifndef LOG_HPP
#define LOG_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <sstream> 

#define SS (std::stringstream{})

namespace L{
    void log(const std::string& message,const std::string& filename = "log.txt")
    {
        std::ofstream file(filename, std::ios::app);
        
        if(file.is_open())
        {
            file << message;
            file.close();
        }
        else{
            std::cerr << "Unable to open log file : " << filename << std::endl;
        }
    }
    void plog(std::stringstream msg,bool overwrite = false,bool logf=true)
    {
        if(overwrite){
            std::cout <<"\r" << msg.str() << std::flush;
        }
        else{
            std::cout << msg.str();
        }
        if(logf)
        {
            log(msg.str());
        }
        
    }
}

#endif