#ifndef HELPER_H
#define HELPER_H

#import <Foundation/Foundation.h>
#include <iostream>
#include <string>

@interface SupObject : NSObject
{
    std::string *name;
}
@end 

#endif