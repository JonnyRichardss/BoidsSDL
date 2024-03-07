#ifndef USE_GAMEMATH
#define USE_GAMEMATH
#include <math.h>
struct Vector2 {
    float x;
    float y;
    Vector2() :x(0), y(0) {}
    Vector2(float _x, float _y) : x(_x), y(_y) {}
    Vector2(float _val) : x(_val), y(_val) {}
    Vector2& operator+=(const Vector2& _other) {
        this->x += _other.x;
        this->y += _other.y;
        return *this;
    }
    Vector2& operator-=(const Vector2& _other) {
        this->x -= _other.x;
        this->y -= _other.y;
        return *this;
    }
    friend Vector2 operator+(Vector2 lhs, const Vector2& rhs) {
        lhs += rhs;
        return lhs;
    }
    friend Vector2 operator-(Vector2 lhs, const Vector2& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Vector2& operator*= (const float& other) {
        this->x *= other;
        this->y *= other;
        return *this;
    }
    friend Vector2 operator* (const float& lhs, Vector2 rhs) {
        return rhs *= lhs;
    }
    friend Vector2 operator* (Vector2 lhs, const float& rhs) {
        return lhs *= rhs;
    }
    static Vector2 one() {
        return Vector2(1, 1);
    }
    static Vector2 up() {
        return Vector2(0, 1);
    }
    static Vector2 right() {
        return Vector2(1, 0);
    }
    static Vector2 zero() {
        return Vector2();
    }
    static float dot(Vector2 a, Vector2 b) {
        return (a.x * b.x) + (a.y * b.y);
    }
    static float AngleBetweenRAD(Vector2 a, Vector2 b) {
        //https://stackoverflow.com/questions/14066933/direct-way-of-computing-the-clockwise-angle-between-two-vectors
        float det = a.x * b.y - a.y * b.x;
        return (atan2(det, dot(a, b)));
    }
    float GetMagnitude() {
        return sqrt(pow(x, 2) + pow(y, 2));
    }
    Vector2 Normalise() {
        float mag = GetMagnitude();
        float _x = x / mag;
        float _y = y / mag;
        return Vector2(_x, _y);
    }
    
};
long double Lerp(long double a, long double b, long double fac);
#endif // !USE_GAMEMATH


