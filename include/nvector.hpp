#pragma once

#include <stdint.h>
#include <string>

struct NVector {
    public:
    uint8_t num_dimensions;
    float *data;

    NVector(uint8_t num_dimensions, float initial_value);
    NVector(uint8_t num_dimensions, float *values);

    NVector operator+(const NVector& other);
    NVector operator-(const NVector& other);
    float& operator[](const uint8_t index);
    float dot(const NVector& other);
    float distance_to(const NVector& other);

    float magnitude();
    void normalize();
    NVector normalize_clone();

    std::string to_string();
    std::string to_csv_string();

    ~NVector();
};