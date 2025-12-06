from solarsynth.synth2 import generate_solar_actual, generate_solar_forecast, generate_load_demand, generate_battery_soc, generate_final_microgrid_dataset
def main():
    generate_solar_actual()
    generate_solar_forecast()
    generate_load_demand()
    generate_battery_soc()
    generate_final_microgrid_dataset()
if __name__ == "__main__":
    main()