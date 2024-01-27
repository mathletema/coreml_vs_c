#include <iostream>
#include <ctime>
#include <CoreML/CoreML.h>

MLModel * model;

void handle_errors(NSError *error) {
    if (error != nil) {
        NSString *formatted = [NSString stringWithFormat:@"%@", [error userInfo]];
        throw std::runtime_error([formatted UTF8String]);
    }
}

MLMultiArray * create_array () {
    NSError *error = nil;

    int N = 3000;


    // Create a 2D multiarray with dimension 3 x 3.
    NSArray<NSNumber *> *shape3x3 = @[@3000, @3000];


    MLMultiArray *multiarray3x3 = [[MLMultiArray alloc] initWithShape:shape3x3 dataType:MLMultiArrayDataTypeFloat error: &error];
    handle_errors(error);

    // Initialize the multiarray.
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            NSNumber *xSubscript = [NSNumber numberWithInt:x];
            NSNumber *ySubscript = [NSNumber numberWithInt:y];


            [multiarray3x3 setObject:@3.14159
                forKeyedSubscript:@[xSubscript, ySubscript]];
        }
    }

    return multiarray3x3;
}

int main () {
    @autoreleasepool {
        NSError *error = nil;

        NSURL *specUrl = [NSURL URLWithString:@"../model.mlpackage"];
        NSURL *compiledUrl = [MLModel compileModelAtURL:specUrl error:&error];

        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsAll;

        MLModel * model = [MLModel modelWithContentsOfURL:compiledUrl configuration:configuration error:&error];
        handle_errors(error);

        std::cout << "making one!\n";

        MLMultiArray * A = create_array();

        std::cout << "making two!\n";

        MLMultiArray * B = create_array();

        NSDictionary<NSString *, id> *featureDictionary = @{
            @"A": A,
            @"B": B,
        };

        std::cout << " starting\n";

        MLDictionaryFeatureProvider *inFeatures = [[MLDictionaryFeatureProvider alloc] initWithDictionary:featureDictionary error:&error];
        handle_errors(error);

        double total_time = 0;
        for (int i = 0; i < 100; i++) {
            clock_t start = clock();
            id<MLFeatureProvider> outFeatures = [model predictionFromFeatures:static_cast<MLDictionaryFeatureProvider * _Nonnull>(inFeatures)
                                                                        error:&error];
            clock_t end = clock();
            double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            total_time += cpu_time_used;
        }

        handle_errors(error);

        std::cout << "Here's some c++ for yall!\n";
        std::cout << "Time taken: " << total_time / 100 << std::endl;
    }
}