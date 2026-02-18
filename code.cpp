#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_HMC5883_U.h>

Adafruit_HMC5883_Unified mag = Adafruit_HMC5883_Unified(12345);

/* ===== Calibration Values (Your Results) ===== */
float offsetX = -0.545;
float offsetY = -5.09;
float offsetZ = -9.34;

float scaleX = 1.152;
float scaleY = 1.029;
float scaleZ = 0.861;
/* ============================================== */

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!mag.begin()) {
    Serial.println("HMC5883L not detected!");
    while (1);
  }

  Serial.println("Compass ready — Magnetic North reference");
}

void loop() {
  sensors_event_t event;
  mag.getEvent(&event);

  // Apply calibration
  float mx = (event.magnetic.x - offsetX) * scaleX;
  float my = (event.magnetic.y - offsetY) * scaleY;
  float mz = (event.magnetic.z - offsetZ) * scaleZ;

  // Heading calculation
  float heading = atan2(my, mx) * 180.0 / PI;

  if (heading < 0)
    heading += 360;

  // Magnetic North reference
  Serial.print("Heading (Magnetic North): ");
  Serial.print(heading);
  Serial.println(" deg");

  delay(200);
}

