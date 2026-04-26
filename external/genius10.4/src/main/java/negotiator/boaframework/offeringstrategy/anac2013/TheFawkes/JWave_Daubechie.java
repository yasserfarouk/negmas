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
 * @author Christian Scheiblich date 26.03.2010 07:35:31 contact graetz@mailfish.de
 */
package negotiator.boaframework.offeringstrategy.anac2013.TheFawkes;

/**
 * Ingrid Daubechies' orthonormal wavelet of eight coefficients and the scales; normed, due to ||*||2 - euclidean norm.
 *
 * @date 26.03.2010 07:35:31
 * @author Christian Scheiblich
 */
public final class JWave_Daubechie extends JWave_Wavelet
{
    /**
     * Constructor setting up the orthonormal Daubechie8 wavelet coeffs and the scales; normed, due to ||*||2 -
     * euclidean norm.
     *
     * @date 26.03.2010 07:35:31
     * @author Christian Scheiblich
     */
    public JWave_Daubechie()
    {
        _waveLength = 8;
        _scales = new double[_waveLength];
        _scales[0] = 0.32580343;
        _scales[1] = 1.01094572;
        _scales[2] = 0.8922014;
        _scales[3] = -0.03967503;
        _scales[4] = -0.2645071;
        _scales[5] = 0.0436163;
        _scales[6] = 0.0465036;
        _scales[7] = -0.01498699;
        // normalize to square root of 2 for being orthonormal
        double sqrt02 = 1.4142135623730951;
        for( int i = 0; i < _waveLength; i++ )
        {
            _scales[i] /= sqrt02;
        }
        _coeffs = new double[_waveLength];
        _coeffs[0] = _scales[7]; //  h7
        _coeffs[1] = -_scales[6]; // -h6
        _coeffs[2] = _scales[5]; //  h5
        _coeffs[3] = -_scales[4]; // -h4
        _coeffs[4] = _scales[3]; //  h3
        _coeffs[5] = -_scales[2]; // -h2
        _coeffs[6] = _scales[1]; //  h1
        _coeffs[7] = -_scales[0]; // -h0
    }
}
