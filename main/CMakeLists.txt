idf_component_register(SRCS "main.c"
                            "audio_ai.cpp"
                            "event_fusion.c"
                            "model_data.c"
                    INCLUDE_DIRS "."
                    REQUIRES nvs_flash
                             esp_driver_i2s
                             esp_driver_gpio
                             esp_timer
                             esp-tflite-micro
                             esp-dsp)
