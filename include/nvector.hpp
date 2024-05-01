#pragma once

#include <stdint.h>
#include <string>

struct NVector {
    public:
    uint8_t num_dimensions;
    float *data;

    NVector(): num_dimensions(0), data(nullptr) {};
    NVector(uint8_t num_dimensions, float initial_value);
    NVector(uint8_t num_dimensions, float *values);
    NVector(const NVector& other);

    NVector operator+(const NVector& other) const;
    void operator+=(const NVector& other);
    NVector operator-(const NVector& other) const;
    void operator/=(const float scalar);
    float& operator[](const uint8_t index) const;
    float dot(const NVector& other) const;
    float distance_to(const NVector& other) const;

    float magnitude() const;
    void normalize();
    NVector normalize_clone() const;

    std::string to_string() const;
    std::string to_csv_string() const;

    ~NVector();
};