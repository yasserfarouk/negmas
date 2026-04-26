package negotiator.boaframework.offeringstrategy.anac2013.TheFawkes;

/*
 * Class:        Polynomial
 * Description:
 * Environment:  Java
 * Software:     SSJ
 * Copyright (C) 2001  Pierre L'Ecuyer and UniversitÃ© de MontrÃ©al
 * Organization: DIRO, UniversitÃ© de MontrÃ©al
 * @author       Ã‰ric Buist

 * SSJ is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (GPL) as published by the
 * Free Software Foundation, either version 3 of the License, or
 * any later version.

 * SSJ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * A copy of the GNU General Public License is available at
 <a href="http://www.gnu.org/licenses">GPL licence site</a>.
 */
/**
 * Represents a polynomial of degree <SPAN CLASS="MATH"><I>n</I></SPAN> in power form. Such a polynomial is of the form
 *
 * <P></P> <DIV ALIGN="CENTER" CLASS="mathdisplay"> <I>p</I>(<I>x</I>) = <I>c</I><SUB>0</SUB> +
 * <I>c</I><SUB>1</SUB><I>x</I> + <SUP> ... </SUP> + <I>c</I><SUB>n</SUB><I>x</I><SUP>n</SUP>, </DIV><P></P> where <SPAN
 * CLASS="MATH"><I>c</I><SUB>0</SUB>,&#8230;, <I>c</I><SUB>n</SUB></SPAN> are the coefficients of the polynomial.
 *
 */
public final class SmoothingPolynomial
{
    private double[] coeff;

    /**
     * Constructs a new polynomial with coefficients <TT>coeff</TT>. The value of <TT>coeff[i]</TT> in this array
     * corresponds to <SPAN CLASS="MATH"><I>c</I><SUB>i</SUB></SPAN>.
     *
     * @param coeff the coefficients of the polynomial.
     *
     * @exception NullPointerException     if <TT>coeff</TT> is <TT>null</TT>.
     *
     * @exception IllegalArgumentException if the length of <TT>coeff</TT> is 0.
     *
     */
    public SmoothingPolynomial( double... coeff )
    {
        if( coeff == null )
        {
            throw new NullPointerException();
        }
        else if( coeff.length == 0 )
        {
            throw new IllegalArgumentException( "At least one coefficient is needed" );
        }
        else
        {
            this.coeff = coeff.clone();
        }
    }

    /**
     * Returns the <SPAN CLASS="MATH"><I>i</I></SPAN>th coefficient of the polynomial.
     *
     * @return the array of coefficients.
     *
     */
    public double getCoefficient( int i )
    {
        return this.coeff[i];
    }

    /**
     * Sets the array of coefficients of this polynomial to <TT>coeff</TT>.
     *
     * @param coeff the new array of coefficients.
     *
     * @exception NullPointerException     if <TT>coeff</TT> is <TT>null</TT>.
     *
     * @exception IllegalArgumentException if the length of <TT>coeff</TT> is 0.
     *
     */
    public void setCoefficients( double... coeff )
    {
        if( coeff == null )
        {
            throw new NullPointerException();
        }
        else if( coeff.length == 0 )
        {
            throw new IllegalArgumentException( "At least one coefficient is needed" );
        }
        else
        {
            this.coeff = coeff.clone();
        }
    }

    public double evaluate( double x )
    {
        double res = this.coeff[this.coeff.length - 1];
        for( int i = ( this.coeff.length - 2 ); i >= 0; i-- )
        {
            res = this.coeff[i] + x * res;
        }
        return res;
    }

    private double getCoeffDer( int i, int n )
    {
        double coeffDer = this.coeff[i];
        for( int j = i; j > ( i - n ); j-- )
        {
            coeffDer *= j;
        }
        return coeffDer;
    }

    public double derivative( double x )
    {
        return this.derivative( x, 1 );
    }

    public double derivative( double x, int n )
    {
        if( n < 0 )
        {
            throw new IllegalArgumentException( "n < 0" );
        }
        else if( n == 0 )
        {
            return this.evaluate( x );
        }
        else if( n >= coeff.length )
        {
            return 0;
        }
        else
        {
            double res = this.getCoeffDer( this.coeff.length - 1, n );
            for( int i = ( this.coeff.length - 2 ); i >= n; i-- )
            {
                res = this.getCoeffDer( i, n ) + ( x * res );
            }
            return res;
        }
    }

    @Override
    public SmoothingPolynomial clone()
    {
        try
        {
            SmoothingPolynomial pol = (SmoothingPolynomial)super.clone();
            pol.setCoefficients( this.coeff );
            return pol;
        }
        catch( CloneNotSupportedException cne )
        {
            throw new IllegalStateException( "Clone not supported (" + cne.getMessage() + ")" );
        }
    }
}
