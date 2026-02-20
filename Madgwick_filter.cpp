#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_HMC5883_U.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;
Adafruit_HMC5883_Unified mag = Adafruit_HMC5883_Unified(12345);

/* ===== MPU6050 Calibration ===== */
float gyroOffsetX = -0.055507;
float gyroOffsetY =  0.004724;
float gyroOffsetZ = -0.017956;

float accOffsetX =  0.075;
float accOffsetY = -0.215;
float accOffsetZ = -1.435;
float accScaleX  =  0.987;
float accScaleY  =  0.948;
float accScaleZ  =  0.924;

/* ===== HMC5883L Calibration ===== */
float magOffsetX = -0.545;
float magOffsetY = -5.09;
float magOffsetZ = -9.34;
float magScaleX  =  1.152;
float magScaleY  =  1.029;
float magScaleZ  =  0.861;

/* ===== Madgwick Filter ===== */
float beta = 0.1f;          // Filter gain — increase for faster response, decrease for less noise
float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;  // Quaternion

unsigned long lastTime = 0;

/* ================================================
   Madgwick AHRS update (accel + gyro + mag)
   All inputs:
     gx,gy,gz  → rad/s  (calibrated)
     ax,ay,az  → any unit, will be normalised
     mx,my,mz  → any unit, will be normalised
   ================================================ */
void MadgwickAHRSupdate(float gx, float gy, float gz,
                         float ax, float ay, float az,
                         float mx, float my, float mz,
                         float dt) {

  float recipNorm;
  float s0, s1, s2, s3;
  float qDot0, qDot1, qDot2, qDot3;
  float hx, hy;
  float _2q0mx, _2q0my, _2q0mz, _2q1mx;
  float _2bx, _2bz;
  float _4bx, _4bz;
  float _2q0, _2q1, _2q2, _2q3;
  float _2q0q2, _2q2q3;
  float q0q0, q0q1, q0q2, q0q3, q1q1, q1q2, q1q3, q2q2, q2q3, q3q3;

  // Rate of change of quaternion from gyroscope
  qDot0 = 0.5f * (-q1*gx - q2*gy - q3*gz);
  qDot1 = 0.5f * ( q0*gx + q2*gz - q3*gy);
  qDot2 = 0.5f * ( q0*gy - q1*gz + q3*gx);
  qDot3 = 0.5f * ( q0*gz + q1*gy - q2*gx);

  // Compute feedback only if accelerometer measurement is valid
  if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {

    // Normalise accelerometer measurement
    recipNorm = 1.0f / sqrt(ax*ax + ay*ay + az*az);
    ax *= recipNorm; ay *= recipNorm; az *= recipNorm;

    // Normalise magnetometer measurement
    recipNorm = 1.0f / sqrt(mx*mx + my*my + mz*mz);
    mx *= recipNorm; my *= recipNorm; mz *= recipNorm;

    // Auxiliary variables
    _2q0mx = 2.0f * q0 * mx;
    _2q0my = 2.0f * q0 * my;
    _2q0mz = 2.0f * q0 * mz;
    _2q1mx = 2.0f * q1 * mx;
    _2q0  = 2.0f * q0;
    _2q1  = 2.0f * q1;
    _2q2  = 2.0f * q2;
    _2q3  = 2.0f * q3;
    _2q0q2 = 2.0f * q0 * q2;
    _2q2q3 = 2.0f * q2 * q3;
    q0q0 = q0*q0; q0q1 = q0*q1; q0q2 = q0*q2; q0q3 = q0*q3;
    q1q1 = q1*q1; q1q2 = q1*q2; q1q3 = q1*q3;
    q2q2 = q2*q2; q2q3 = q2*q3; q3q3 = q3*q3;

    // Reference direction of Earth's magnetic field in world frame
    hx = mx*(q0q0+q1q1-q2q2-q3q3) + 2.0f*my*(q1q2-q0q3) + 2.0f*mz*(q1q3+q0q2);
    hy = 2.0f*mx*(q1q2+q0q3) + my*(q0q0-q1q1+q2q2-q3q3) + 2.0f*mz*(q2q3-q0q1);
    _2bx = sqrt(hx*hx + hy*hy);
    _2bz = -2.0f*mx*(q1q3-q0q2) + 2.0f*my*(q2q3+q0q1) + mz*(q0q0-q1q1-q2q2+q3q3);
    _4bx = 2.0f * _2bx;
    _4bz = 2.0f * _2bz;

    // Gradient descent algorithm corrective step
    s0 = -_2q2*(2.0f*(q1q3-q0q2)-ax)
         + _2q1*(2.0f*(q0q1+q2q3)-ay)
         - _2bz*q2*(_2bx*(0.5f-q2q2-q3q3) + _2bz*(q1q3-q0q2) - mx)
         + (-_2bx*q3+_2bz*q1)*(_2bx*(q1q2-q0q3)+_2bz*(q0q1+q2q3)-my)
         + _2bx*q2*(_2bx*(q1q3+q0q2)+_2bz*(q2q3-q0q1)-mz);

    s1 = _2q3*(2.0f*(q1q3-q0q2)-ax)
         + _2q0*(2.0f*(q0q1+q2q3)-ay)
         - 4.0f*q1*(1.0f-2.0f*(q1q1+q2q2)-az)
         + _2bz*q3*(_2bx*(0.5f-q2q2-q3q3)+_2bz*(q1q3-q0q2)-mx)
         + (_2bx*q2+_2bz*q0)*(_2bx*(q1q2-q0q3)+_2bz*(q0q1+q2q3)-my)
         + (_2bx*q3-_4bz*q1)*(_2bx*(q1q3+q0q2)+_2bz*(q2q3-q0q1)-mz);

    s2 = -_2q0*(2.0f*(q1q3-q0q2)-ax)
         + _2q3*(2.0f*(q0q1+q2q3)-ay)
         - 4.0f*q2*(1.0f-2.0f*(q1q1+q2q2)-az)
         + (-_4bx*q2-_2bz*q0)*(_2bx*(0.5f-q2q2-q3q3)+_2bz*(q1q3-q0q2)-mx)
         + (_2bx*q1+_2bz*q3)*(_2bx*(q1q2-q0q3)+_2bz*(q0q1+q2q3)-my)
         + (_2bx*q0-_4bz*q2)*(_2bx*(q1q3+q0q2)+_2bz*(q2q3-q0q1)-mz);

    s3 = _2q1*(2.0f*(q1q3-q0q2)-ax)
         + _2q2*(2.0f*(q0q1+q2q3)-ay)
         + (-_4bx*q3+_2bz*q1)*(_2bx*(0.5f-q2q2-q3q3)+_2bz*(q1q3-q0q2)-mx)
         + (-_2bx*q0+_2bz*q2)*(_2bx*(q1q2-q0q3)+_2bz*(q0q1+q2q3)-my)
         + _2bx*q1*(_2bx*(q1q3+q0q2)+_2bz*(q2q3-q0q1)-mz);

    // Normalise step magnitude
    recipNorm = 1.0f / sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3);
    s0 *= recipNorm; s1 *= recipNorm; s2 *= recipNorm; s3 *= recipNorm;

    // Apply feedback
    qDot0 -= beta * s0;
    qDot1 -= beta * s1;
    qDot2 -= beta * s2;
    qDot3 -= beta * s3;
  }

  // Integrate to yield quaternion
  q0 += qDot0 * dt;
  q1 += qDot1 * dt;
  q2 += qDot2 * dt;
  q3 += qDot3 * dt;

  // Normalise quaternion
  recipNorm = 1.0f / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
  q0 *= recipNorm; q1 *= recipNorm; q2 *= recipNorm; q3 *= recipNorm;
}

/* ================================================ */

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!"); while (1);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  if (!mag.begin()) {
    Serial.println("HMC5883L not found!"); while (1);
  }

  Serial.println("MPU6050 + HMC5883L ready — Madgwick AHRS");
  lastTime = micros();
}

void loop() {
  // ---- Compute dt ----
  unsigned long now = micros();
  float dt = (now - lastTime) / 1000000.0f;
  lastTime = now;

  // ---- Read MPU6050 ----
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  float ax = (accel.acceleration.x - accOffsetX) * accScaleX;
  float ay = (accel.acceleration.y - accOffsetY) * accScaleY;
  float az = (accel.acceleration.z - accOffsetZ) * accScaleZ;

  float gx = gyro.gyro.x - gyroOffsetX;   // rad/s
  float gy = gyro.gyro.y - gyroOffsetY;
  float gz = gyro.gyro.z - gyroOffsetZ;

  // ---- Read HMC5883L ----
  sensors_event_t magEvent;
  mag.getEvent(&magEvent);

  float mx = (magEvent.magnetic.x - magOffsetX) * magScaleX;
  float my = (magEvent.magnetic.y - magOffsetY) * magScaleY;
  float mz = (magEvent.magnetic.z - magOffsetZ) * magScaleZ;

  // ---- Madgwick Filter ----
  MadgwickAHRSupdate(gx, gy, gz, ax, ay, az, mx, my, mz, dt);

  // ---- Quaternion → Roll / Pitch / Yaw ----
  float roll  = atan2(2.0f*(q0*q1 + q2*q3),
                      1.0f - 2.0f*(q1*q1 + q2*q2)) * 180.0f / PI;

  float pitchArg = 2.0f*(q0*q2 - q3*q1);
  pitchArg = constrain(pitchArg, -1.0f, 1.0f);
  float pitch = asin(pitchArg) * 180.0f / PI;

  float yaw   = atan2(2.0f*(q0*q3 + q1*q2),
                      1.0f - 2.0f*(q2*q2 + q3*q3)) * 180.0f / PI;
  if (yaw < 0) yaw += 360.0f;   // 0–360° magnetic north reference

  // ---- Print ----
  Serial.print("Roll: ");   Serial.print(roll,  2);
  Serial.print("  Pitch: "); Serial.print(pitch, 2);
  Serial.print("  Yaw: ");  Serial.println(yaw,  2);

  delay(10);   // ~100 Hz
}
