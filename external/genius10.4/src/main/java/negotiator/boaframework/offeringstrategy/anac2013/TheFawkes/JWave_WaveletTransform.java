package negotiator.boaframework.offeringstrategy.anac2013.TheFawkes;

/**
 * @date 30 juin 2011 14:50:35
 * @author Pol Kennel
 */
public abstract class JWave_WaveletTransform
{
    protected final JWave_Wavelet _wavelet;
    protected final int _iteration;

    public JWave_WaveletTransform( JWave_Wavelet wavelet )
    {
        this._wavelet = wavelet;
        this._iteration = -1;
    }

    public JWave_WaveletTransform( JWave_Wavelet wavelet, int iteration )
    {
        this._wavelet = wavelet;
        this._iteration = iteration;
    }

    public double[] forward( double[] arrTime )
    {
        if( this._iteration <= 0 )
        {
            return forwardWavelet( arrTime );
        }
        else
        {
            return forwardWavelet( arrTime, this._iteration );
        }
    }

    public double[] reverse( double[] arrFreq )
    {
        if( _iteration <= 0 )
        {
            return reverseWavelet( arrFreq );
        }
        else
        {
            return reverseWavelet( arrFreq, this._iteration );
        }
    }

    public abstract double[] forwardWavelet( double[] arrTime );

    /**
     * Performs the forward transform from time domain to frequency or Hilbert domain for a given array depending on the
     * used transform algorithm by inheritance. The number of transformation levels applied is limited by threshold.
     *
     * @date 15.07.2010
     * @author Thomas Haider
     * @date 15.08.2010 00:32:09
     * @author Christian Scheiblich
     * @param arrTime coefficients of 1-D time domain
     * @param toLevel threshold for number of iterations
     * @return coefficients of 1-D frequency or Hilbert domain
     */
    public abstract double[] forwardWavelet( double[] arrTime, int toLevel );

    public abstract double[] reverseWavelet( double[] arrTime );

    /**
     * Performs the reverse transform from frequency or Hilbert domain to time domain for a given array depending on the
     * used transform algorithm by inheritance. The number of transformation levels applied is limited by threshold.
     *
     * @date 15.07.2010
     * @author Thomas Haider
     * @date 15.08.2010 00:32:24
     * @author Christian Scheiblich
     * @param arrFreq   coefficients of 1-D frequency or Hilbert domain
     * @param fromLevel threshold for number of iterations
     * @return coefficients of 1-D time domain
     */
    public abstract double[] reverseWavelet( double[] arrFreq, int fromLevel );
}
