package genius.core.misc;

import static java.lang.Math.pow;
import static java.lang.String.format;


/**
 * Contains helper functions for time
 */
public class Time {

    /** a day in seconds */
    public static final int DAY = 86400;

    /** an hour in seconds */
    public static final int HOUR = 3600;

    /** a minute in seconds */
    public static final int MINUTE = 60;

    /**
     * Converts the given nano time to an human readable string.
     * for example prettyTimeSpan(n) = 5 hours, 2 minutes, 3 seconds
     * @param nanoTime The time to convert
     * @return The time string.
     */
    public static String prettyTimeSpan(double nanoTime) {
        return prettyTimeSpan(nanoTime, true);
    }

    /**
     * Converts the given nano time to an human readable string.
     * for example prettyTimeSpan(n) = 5 hours, 2 minutes, 3 seconds
     * @param nanoTime The time to convert
     * @param showSeconds if set to true, round to seconds. if set to false, round to minutes
     * @return The time string.
     */
    public static String prettyTimeSpan(double nanoTime, boolean showSeconds) {
        if (Double.isInfinite(nanoTime) || Double.isNaN(nanoTime)) {
            return "Unknown";
        }
        int t = (int) Math.floor(nanoTime / pow(10, 9));
        String prettyTimeSpan = "";

        int days = t / DAY;
        int hours = t % DAY / HOUR;
        int minutes = t % DAY % HOUR / MINUTE;
        int seconds = t % DAY % HOUR % MINUTE;

        if (days == 1)
            prettyTimeSpan += format("%d day, ", days);
        if (days > 1)
            prettyTimeSpan += format("%d days, ", days);
        if (hours == 1)
            prettyTimeSpan += format("%d hour, ", hours);
        if (hours > 1)
            prettyTimeSpan += format("%d hours, ", hours);
        if (minutes == 1)
            prettyTimeSpan += format("%d minute", minutes);
        if (minutes > 1)
            prettyTimeSpan += format("%d minutes", minutes);
        if (minutes >= 1 && showSeconds)
            prettyTimeSpan += ", ";
        if (seconds == 1 && showSeconds)
            prettyTimeSpan += format("%d second", seconds);
        if (seconds > 1 && showSeconds)
            prettyTimeSpan += format("%d seconds", seconds);
        if (prettyTimeSpan.isEmpty())
            prettyTimeSpan = "< 1 minute";

        return prettyTimeSpan;
    }
}
