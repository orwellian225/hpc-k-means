#include <cmath>
#include <stdint.h>
#include <string>

#include <fmt/core.h>

#include "nvector.hpp"

NVector::NVector(uint8_t num_dimensions, float initial_value) {
    this->num_dimensions = num_dimensions;
    this->data = new float[num_dimensions];

    for (uint8_t d = 0; d < num_dimensions; ++d)
        this->data[d] = initial_value;
}

NVector::NVector(uint8_t num_dimensions, float *values) {
    this->num_dimensions = num_dimensions;
    this->data = values;
}

NVector NVector::operator+(const NVector& other) const {
    NVector result(num_dimensions, 0.0);

    for (uint8_t d = 0; d < num_dimensions; ++d)
        result.data[d] = this->data[d] + other.data[d];

    return result;
}

NVector NVector::operator-(const NVector& other) const {
    NVector result(num_dimensions, 0.0);

    for (uint8_t d = 0; d < num_dimensions; ++d)
        result.data[d] = this->data[d] - other.data[d];

    return result;
}

float& NVector::operator[](const uint8_t index) const {
    return data[index];
}

float NVector::dot(const NVector& other) const {
    float result;

    for (uint8_t d = 0; d < num_dimensions; ++d)
        result += this->data[d] * other.data[d];

    return result;
}

float NVector::distance_to(const NVector& other) const {
    float result;

    for (uint8_t d = 0; d < num_dimensions; ++d)
        result += (this->data[d] - other.data[d]) * (this->data[d] - other.data[d]);

    return std::sqrt(result);
}

float NVector::magnitude() const {
    float result;

    for (uint8_t d = 0; d < num_dimensions; ++d)
        result += data[d] * data[d];

    return std::sqrt(result);
}

void NVector::normalize() {
    float magnitude = this->magnitude();

    for (uint8_t d = 0; d < num_dimensions; ++d)
        data[d] /= magnitude;
}

NVector NVector::normalize_clone() const {
    NVector result(num_dimensions, 0.0);
    float magnitude = this->magnitude();

    for (uint8_t d = 0; d < num_dimensions; ++d)
        result.data[d] = data[d] / magnitude;

    return result;
}

std::string NVector::to_string() const {
    std::string result = fmt::format("{}", data[0]);
    for (uint8_t d = 1; d < num_dimensions; ++d)
        result = fmt::format("{}, {}", result, data[d]);

    return result;
}

std::string NVector::to_csv_string() const {
    std::string result = fmt::format("{}", data[0]);
    for (uint8_t d = 1; d < num_dimensions; ++d)
        result = fmt::format("{},{}", result, data[d]);

    return result;
}

NVector::~NVector() {
    delete[] data;
}