package negotiator.boaframework.offeringstrategy.anac2013.TheFawkes;

public final class Fawkews_Math
{
    public static double getMean( double[] values )
    {
        double mean = 0;
        if( values != null && values.length > 0 )
        {
            for( double value : values )
            {
                mean += value;
            }
            mean /= values.length;
        }
        return mean;
    }

    public static double getStandardDeviation( double[] values )
    {
        double deviation = 0;
        if( values != null && values.length > 1 )
        {
            double mean = getMean( values );
            for( double value : values )
            {
                double delta = value - mean;
                deviation += delta * delta;
            }
            deviation = Math.sqrt( deviation / values.length );
        }
        return deviation;
    }
}
