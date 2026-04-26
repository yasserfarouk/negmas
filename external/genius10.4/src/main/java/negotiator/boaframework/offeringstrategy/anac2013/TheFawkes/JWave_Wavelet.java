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
 * Basic class for one wavelet keeping coefficients of the wavelet function, the scaling function, the base wavelength,
 * the forward transform method, and the reverse transform method.
 *
 * @date 10.02.2010 08:54:48
 * @author Christian Scheiblich
 */
public abstract class JWave_Wavelet
{
    /**
     * minimal wavelength of the used wavelet and scaling coefficients
     */
    protected int _waveLength;
    /**
     * coefficients of the wavelet; wavelet function
     */
    protected double[] _coeffs;
    /**
     * coefficients of the scales; scaling function
     */
    protected double[] _scales;

    /**
     * Constructor; predefine members to init values
     *
     * @date 10.02.2010 08:54:48
     * @author Christian Scheiblich
     */
    public JWave_Wavelet()
    {
        this._waveLength = 0;
        this._coeffs = null;
        this._scales = null;
    }

    /**
     * Performs the forward transform for the given array from time domain to Hilbert domain and returns a new array of
     * the same size keeping coefficients of Hilbert domain and should be of length 2 to the power of p -- length = 2^p
     * where p is a positive integer.
     *
     * @date 10.02.2010 08:18:02
     * @author Christian Scheiblich
     * @param arrTime array keeping time domain coefficients
     * @return coefficients represented by frequency domain
     */
    public double[] forward( double[] arrTime )
    {
        double[] arrHilb = new double[arrTime.length];
        int k, h = arrTime.length >> 1;
        for( int i = 0; i < h; i++ )
        {
            for( int j = 0; j < _waveLength; j++ )
            {
                k = ( i << 1 ) + j;
                while( k >= arrTime.length )
                {
                    k -= arrTime.length;
                }
                arrHilb[i] += arrTime[k] * this._scales[j]; // low pass filter - energy (approximation)
                arrHilb[i + h] += arrTime[k] * this._coeffs[j]; // high pass filter - details
            }
        }
        return arrHilb;
    }

    /**
     * Performs the reverse transform for the given array from Hilbert domain to time domain and returns a new array of
     * the same size keeping coefficients of time domain and should be of length 2 to the power of p -- length = 2^p
     * where p is a positive integer.
     *
     * @date 10.02.2010 08:19:24
     * @author Christian Scheiblich
     * @param arrHilb array keeping frequency domain coefficients
     * @return coefficients represented by time domain
     */
    public double[] reverse( double[] arrHilb )
    {
        double[] arrTime = new double[arrHilb.length];
        int k, h = arrHilb.length >> 1;
        for( int i = 0; i < h; i++ )
        {
            for( int j = 0; j < _waveLength; j++ )
            {
                k = ( i << 1 ) + j;
                while( k >= arrHilb.length )
                {
                    k -= arrHilb.length;
                }
                arrTime[k] += ( arrHilb[i] * this._scales[j] ) + ( arrHilb[i + h] * this._coeffs[j] ); // adding up details times energy (approximation)
            }
        }
        return arrTime;
    }

    /**
     * Returns the minimal wavelength for the used wavelet.
     *
     * @date 10.02.2010 08:13:59
     * @author Christian Scheiblich
     * @return the minimal wavelength for this basic wave
     */
    public int getWaveLength()
    {
        return this._waveLength;
    }
}
