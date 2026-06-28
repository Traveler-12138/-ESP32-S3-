idf_component_register(SRCS "mic.c" "main.c"
                            "audio_ai.cpp"
                            "event_fusion.c"
                            "model_data.c"
                    INCLUDE_DIRS "."
                    REQUIRES nvs_flash
                             driver
                             esp_driver_gpio
                             esp_timer
                             esp-tflite-micro
                             esp-dsp)
