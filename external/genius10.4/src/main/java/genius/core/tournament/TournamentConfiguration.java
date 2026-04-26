package genius.core.tournament;

import java.util.HashMap;

public class TournamentConfiguration {

	private static HashMap<String, Integer> configuration;
	
	public static void setConfiguration(HashMap<String, Integer> config) {
		configuration = config;
	}
	
	public static void addOption(String option, int value) {
		if (configuration == null) {
			configuration = new HashMap<String, Integer>();
		}
		configuration.put(option, value);
	}
	
	public static boolean getBooleanOption(String option, boolean defaultSetting) {
		boolean result = defaultSetting;
		if (configuration != null && configuration.containsKey(option)) {
			result = configuration.get(option) != 0;
		} else {
//			System.err.println("NegotiationConfiguration for " + option + " not set. Using default: " + defaultSetting + ".");
		}
		return result;
	}
	
	public static int getIntegerOption(String option, int defaultSetting) {
		int result = defaultSetting;
		if (configuration != null && configuration.containsKey(option)) {
			result = configuration.get(option);
		} else {
			System.err.println("NegotiationConfiguration for " + option + " not set. Using default: " + defaultSetting + ".");
		}
		return result;
	}

	public static HashMap<String, Integer> getOptions() {
		return configuration;
	}
}
