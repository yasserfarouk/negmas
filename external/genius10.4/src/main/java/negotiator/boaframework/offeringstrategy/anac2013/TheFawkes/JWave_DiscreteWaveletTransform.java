/**
 * JWave - Java implementation of wavelet transform algorithms
 *
 * Copyright 2010-2012 Christian Scheiblich
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * @author Christian Scheiblich date 23.02.2010 05:42:23 contact graetz@mailfish.de
 */
package negotiator.boaframework.offeringstrategy.anac2013.TheFawkes;

/**
 * Base class for the forward and reverse Discret JWave_Wavelet Transform in 1-D using a specified JWave_Wavelet by
 * inheriting class.
 *
 * @date 23 juin 2011 15:55:33
 * @author Pol Kennel
 */
public final class JWave_DiscreteWaveletTransform extends JWave_WaveletTransform
{
    /**
     * Constructor receiving a WaveletI object.
     *
     * @date 23 juin 2011 15:54:54
     * @author Pol Kennel
     * @param wavelet
     */
    public JWave_DiscreteWaveletTransform( JWave_Wavelet wavelet )
    {
        super( wavelet );
    }

    /**
     * Constructor receiving a WaveletI object and a iteration level for forward reverse methods
     *
     * @date 23 juin 2011 15:54:54
     * @author Pol Kennel
     * @param wavelet
     */
    public JWave_DiscreteWaveletTransform( JWave_Wavelet wavelet, int iteration )
    {
        super( wavelet, iteration );
    }

    /**
     * Performs the 1-D forward transform for arrays of dim N from time domain to Hilbert domain for the given array
     * using the Discrete JWave_Wavelet Transform (DWT) algorithm (identical to the Fast JWave_Wavelet Transform (FWT)
     * in 1-D).
     *
     * @date 24 juin 2011 13:16:00
     * @author Pol Kennel
     * @date 10.02.2010 08:23:24
     * @author Christian Scheiblich
     * @param arrTime coefficients of 1-D Time domain
     */
    @Override
    public double[] forwardWavelet( double[] arrTime )
    {
        double[] arrHilb = new double[arrTime.length];
        System.arraycopy( arrTime, 0, arrHilb, 0, arrTime.length );
        int h = arrTime.length;
        int minWaveLength = this._wavelet.getWaveLength();
        if( h >= minWaveLength )
        {
            while( h >= minWaveLength )
            {
                double[] iBuf = new double[h];
                System.arraycopy( arrHilb, 0, iBuf, 0, h );
                double[] oBuf = this._wavelet.forward( iBuf );
                System.arraycopy( oBuf, 0, arrHilb, 0, h );
                h >>= 1;
            }
        }
        return arrHilb;
    }

    /**
     * Performs the 1-D reverse transform for arrays of dim N from Hilbert domain to time domain for the given array
     * using the Discrete JWave_Wavelet Transform (DWT) algorithm and the selected wavelet (identical to the Fast
     * JWave_Wavelet Transform (FWT) in 1-D).
     *
     * @date 24 juin 2011 13:16:18
     * @author Pol Kennel
     * @date 10.02.2010 08:23:24
     * @author Christian Scheiblich
     * @param arrHilb coefficients of 1-D Hilbert domain
     */
    @Override
    public double[] reverseWavelet( double[] arrHilb )
    {
        double[] arrTime = new double[arrHilb.length];
        System.arraycopy( arrHilb, 0, arrTime, 0, arrHilb.length );
        int minWaveLength = this._wavelet.getWaveLength();
        int h = minWaveLength;
        if( arrHilb.length >= minWaveLength )
        {
            while( h <= arrTime.length && h >= minWaveLength )
            {
                double[] iBuf = new double[h];
                System.arraycopy( arrTime, 0, iBuf, 0, h );
                double[] oBuf = this._wavelet.reverse( iBuf );
                System.arraycopy( oBuf, 0, arrTime, 0, h );
                h <<= 1;
            }
        }
        return arrTime;
    }

    /**
     * Performs the 1-D forward transform for arrays of dim N from time domain to Hilbert domain for the given array
     * using the Discrete JWave_Wavelet Transform (DWT) algorithm. The number of transformation levels applied is
     * limited by threshold (identical to the Fast JWave_Wavelet Transform (FWT) in 1-D).
     *
     * @date 24 juin 2011 13:18:38
     * @author Pol Kennel
     * @date 15.07.2010 13:26:26
     * @author Thomas Haider
     * @date 15.08.2010 00:31:36
     * @author Christian Scheiblich
     * @param arrTime coefficients of 1-D Time domain
     * @param toLevel iteration number
     */
    @Override
    public double[] forwardWavelet( double[] arrTime, int toLevel )
    {
        double[] arrHilb = new double[arrTime.length];
        System.arraycopy( arrTime, 0, arrHilb, 0, arrTime.length );
        int level = 0;
        int h = arrTime.length;
        int minWaveLength = this._wavelet.getWaveLength();
        if( h >= minWaveLength )
        {
            while( h >= minWaveLength && level < toLevel )
            {
                double[] iBuf = new double[h];
                System.arraycopy( arrHilb, 0, iBuf, 0, h );
                double[] oBuf = this._wavelet.forward( iBuf );
                System.arraycopy( oBuf, 0, arrHilb, 0, h );
                h >>= 1;
                level++;
            }
        }
        return arrHilb;
    }

    /**
     * Performs the 1-D reverse transform for arrays of dim N from Hilbert domain to time domain for the given array
     * using the Discrete JWave_Wavelet Transform (DWT) algorithm and the selected wavelet (identical to the Fast
     * JWave_Wavelet Transform (FWT) in 1-D). The number of transformation levels applied is limited by threshold.
     *
     * @author Pol Kennel
     * @date 24 juin 2011 13:43:05
     * @author Thomas Haider
     * @date 15.08.2010 00:31:09
     * @author Christian Scheiblich
     * @date 20.06.2011 13:03:27
     * @param arrHilb   coefficients of 1-D Hilbert domain
     * @param fromLevel iteration number
     */
    @Override
    public double[] reverseWavelet( double[] arrHilb, int fromLevel )
    {
        double[] arrTime = new double[arrHilb.length];
        System.arraycopy( arrHilb, 0, arrTime, 0, arrHilb.length );
        int level = 0;
        int minWaveLength = this._wavelet.getWaveLength();
        int h = (int)( arrHilb.length / ( Math.pow( 2, fromLevel - 1 ) ) );
        if( arrHilb.length >= minWaveLength )
        {
            while( h <= arrTime.length && h >= minWaveLength && level < fromLevel )
            {
                double[] iBuf = new double[h];
                System.arraycopy( arrTime, 0, iBuf, 0, h );
                double[] oBuf = this._wavelet.reverse( iBuf );
                System.arraycopy( oBuf, 0, arrTime, 0, h );
                h <<= 1;
                level++;
            }
        }
        return arrTime;
    }
}