package negotiator.boaframework.offeringstrategy.anac2013.TheFawkes;

/*
 * Class:        SmoothingCubicSpline
 * Description:  smoothing cubic spline algorithm of Schoenberg
 * Environment:  Java
 * Software:     SSJ
 * Copyright (C) 2001  Pierre L'Ecuyer and UniversitÃ© de MontrÃ©al
 * Organization: DIRO, UniversitÃ© de MontrÃ©al
 * @author

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
 * Represents a cubic spline with nodes at <SPAN CLASS="MATH">(<I>x</I><SUB>i</SUB>, <I>y</I><SUB>i</SUB>)</SPAN>
 * computed with the smoothing cubic spline algorithm of Schoenberg. A smoothing cubic spline is made of <SPAN
 * CLASS="MATH"><I>n</I> + 1</SPAN> cubic polynomials. The <SPAN CLASS="MATH"><I>i</I></SPAN>th polynomial of such a
 * spline, for <SPAN CLASS="MATH"><I>i</I> = 1,&#8230;, <I>n</I> - 1</SPAN>, is defined as <SPAN
 * CLASS="MATH"><I>S</I><SUB>i</SUB>(<I>x</I>)</SPAN> while the complete spline is defined as
 *
 * <P></P> <DIV ALIGN="CENTER" CLASS="mathdisplay"> <I>S</I>(<I>x</I>) =
 * <I>S</I><SUB>i</SUB>(<I>x</I>),&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for
 * <I>x</I>&#8712;[<I>x</I><SUB>i-1</SUB>, <I>x</I><SUB>i</SUB>]. </DIV><P></P> For <SPAN CLASS="MATH"><I>x</I> &lt;
 * <I>x</I><SUB>0</SUB></SPAN> and <SPAN CLASS="MATH"><I>x</I> &gt; <I>x</I><SUB>n-1</SUB></SPAN>, the spline is not
 * precisely defined, but this class performs extrapolation by using <SPAN CLASS="MATH"><I>S</I><SUB>0</SUB></SPAN> and
 * <SPAN CLASS="MATH"><I>S</I><SUB>n</SUB></SPAN> linear polynomials. The algorithm which calculates the smoothing
 * spline is a generalization of the algorithm for an interpolating spline. <SPAN
 * CLASS="MATH"><I>S</I><SUB>i</SUB></SPAN> is linked to <SPAN CLASS="MATH"><I>S</I><SUB>i+1</SUB></SPAN> at <SPAN
 * CLASS="MATH"><I>x</I><SUB>i+1</SUB></SPAN> and keeps continuity properties for first and second derivatives at this
 * point, therefore
 *
 * <SPAN CLASS="MATH"><I>S</I><SUB>i</SUB>(<I>x</I><SUB>i+1</SUB>) =
 * <I>S</I><SUB>i+1</SUB>(<I>x</I><SUB>i+1</SUB>)</SPAN>,
 *
 * <SPAN CLASS="MATH"><I>S'</I><SUB>i</SUB>(<I>x</I><SUB>i+1</SUB>) =
 * <I>S'</I><SUB>i+1</SUB>(<I>x</I><SUB>i+1</SUB>)</SPAN> and <SPAN
 * CLASS="MATH"><I>S''</I><SUB>i</SUB>(<I>x</I><SUB>i+1</SUB>) =
 * <I>S''</I><SUB>i+1</SUB>(<I>x</I><SUB>i+1</SUB>)</SPAN>.
 *
 * <P> The spline is computed with a smoothing parameter <SPAN CLASS="MATH"><I>&#961;</I>&#8712;[0, 1]</SPAN> which
 * represents its accuracy with respect to the initial <SPAN CLASS="MATH">(<I>x</I><SUB>i</SUB>,
 * <I>y</I><SUB>i</SUB>)</SPAN> nodes. The smoothing spline minimizes
 *
 * <P></P> <DIV ALIGN="CENTER" CLASS="mathdisplay"> <I>L</I> =
 * <I>&#961;</I>&sum;<SUB>i=0</SUB><SUP>n-1</SUP><I>w</I><SUB>i</SUB>(<I>y</I><SUB>i</SUB>-<I>S</I><SUB>i</SUB>(<I>x</I><SUB>i</SUB>))<SUP>2</SUP>
 * + (1 -
 * <I>&#961;</I>)&int;<SUB>x<SUB>0</SUB></SUB><SUP>x<SUB>n-1</SUB></SUP>(<I>S''</I>(<I>x</I>))<SUP>2</SUP><I>dx</I>
 * </DIV><P></P> In fact, by setting <SPAN CLASS="MATH"><I>&#961;</I> = 1</SPAN>, we obtain the interpolating spline;
 * and we obtain a linear function by setting <SPAN CLASS="MATH"><I>&#961;</I> = 0</SPAN>. The weights <SPAN
 * CLASS="MATH"><I>w</I><SUB>i</SUB> &gt; 0</SPAN>, which default to 1, can be used to change the contribution of each
 * point in the error term. A large value <SPAN CLASS="MATH"><I>w</I><SUB>i</SUB></SPAN> will give a large weight to the
 * <SPAN CLASS="MATH"><I>i</I></SPAN>th point, so the spline will pass closer to it. Here is a small example that uses
 * smoothing splines:
 *
 * <P>
 *
 * <DIV CLASS="vcode" ALIGN="LEFT"> <TT>
 *
 * <BR>&nbsp;&nbsp;&nbsp;int n; <BR>&nbsp;&nbsp;&nbsp;double[] X = new double[n]; <BR>&nbsp;&nbsp;&nbsp;double[] Y = new
 * double[n]; <BR>&nbsp;&nbsp;&nbsp;// here, fill arrays X and Y with n data points (x_i, y_i) <BR>&nbsp;&nbsp;&nbsp;//
 * The points must be sorted with respect to x_i. <BR> <BR>&nbsp;&nbsp;&nbsp;double rho = 0.1;
 * <BR>&nbsp;&nbsp;&nbsp;SmoothingCubicSpline fit = new SmoothingCubicSpline(X, Y, rho); <BR> <BR>&nbsp;&nbsp;&nbsp;int
 * m = 40; <BR>&nbsp;&nbsp;&nbsp;double[] Xp = new double[m+1]; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Xp, Yp are spline
 * points <BR>&nbsp;&nbsp;&nbsp;double[] Yp = new double[m+1]; <BR>&nbsp;&nbsp;&nbsp;double h = (X[n-1] - X[0]) / m;
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// step <BR> <BR>&nbsp;&nbsp;&nbsp;for (int i = 0; i &lt;= m; i++) {
 * <BR>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double z = X[0] + i * h; <BR>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Xp[i] = z;
 * <BR>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Yp[i] = fit.evaluate(z);
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// evaluate spline at z <BR>&nbsp;&nbsp;&nbsp;} <BR> <BR></TT>
 * </DIV>
 *
 */
public final class SmoothingCubicSpline
{
    private final SmoothingPolynomial[] splineVector;
    private final double[] x, y, weight;
    private final double rho;

    /**
     * Constructs a spline with nodes at <SPAN CLASS="MATH">(<I>x</I><SUB>i</SUB>, <I>y</I><SUB>i</SUB>)</SPAN>, with
     * weights <SPAN CLASS="MATH"><I>w</I><SUB>i</SUB></SPAN> and smoothing factor <SPAN
     * CLASS="MATH"><I>&#961;</I></SPAN> = <TT>rho</TT>. The <SPAN CLASS="MATH"><I>x</I><SUB>i</SUB></SPAN> <SPAN
     * CLASS="textit">must</SPAN> be sorted in increasing order.
     *
     * @param x   the <SPAN CLASS="MATH"><I>x</I><SUB>i</SUB></SPAN> coordinates.
     *
     * @param y   the <SPAN CLASS="MATH"><I>y</I><SUB>i</SUB></SPAN> coordinates.
     *
     * @param w   the weight for each point, must be <SPAN CLASS="MATH">&gt; 0</SPAN>.
     *
     * @param rho the smoothing parameter
     *
     * @exception IllegalArgumentException if <TT>x</TT>, <TT>y</TT> and <TT>z</TT> do not have the same length, if rho
     *                                     has wrong value, or if the spline cannot be calculated.
     *
     */
    public SmoothingCubicSpline( double[] x, double[] y, double[] w, double rho )
    {
        if( x.length != y.length )
        {
            throw new IllegalArgumentException( "x.length != y.length" );
        }
        else if( w != null && x.length != w.length )
        {
            throw new IllegalArgumentException( "x.length != w.length" );
        }
        else if( rho < 0 || rho > 1 )
        {
            throw new IllegalArgumentException( "rho not in [0, 1]" );
        }
        else
        {
            this.splineVector = new SmoothingPolynomial[x.length + 1];
            this.rho = rho;
            this.x = x.clone();
            this.y = y.clone();
            this.weight = new double[x.length];
            if( w == null )
            {
                for( int i = 0; i < this.weight.length; i++ )
                {
                    this.weight[i] = 1;
                }
            }
            else
            {
                System.arraycopy( w, 0, this.weight, 0, this.weight.length );
            }
            this.resolve();
        }
    }

    /**
     * Constructs a spline with nodes at <SPAN CLASS="MATH">(<I>x</I><SUB>i</SUB>, <I>y</I><SUB>i</SUB>)</SPAN>, with
     * weights <SPAN CLASS="MATH">= 1</SPAN> and smoothing factor <SPAN CLASS="MATH"><I>&#961;</I></SPAN> =
     * <TT>rho</TT>. The <SPAN CLASS="MATH"><I>x</I><SUB>i</SUB></SPAN> <SPAN CLASS="textit">must</SPAN> be sorted in
     * increasing order.
     *
     * @param x   the <SPAN CLASS="MATH"><I>x</I><SUB>i</SUB></SPAN> coordinates.
     *
     * @param y   the <SPAN CLASS="MATH"><I>y</I><SUB>i</SUB></SPAN> coordinates.
     *
     * @param rho the smoothing parameter
     *
     * @exception IllegalArgumentException if <TT>x</TT> and <TT>y</TT> do not have the same length, if rho has wrong
     *                                     value, or if the spline cannot be calculated.
     *
     */
    public SmoothingCubicSpline( double[] x, double[] y, double rho )
    {
        this( x, y, null, rho );
    }

    /**
     * Evaluates and returns the value of the spline at <SPAN CLASS="MATH"><I>z</I></SPAN>.
     *
     * @param z argument of the spline.
     *
     * @return value of spline.
     *
     */
    public double evaluate( double z )
    {
        double returned;
        int i = this.getFitPolynomialIndex( z );
        if( i == 0 )
        {
            returned = this.splineVector[i].evaluate( z - x[0] );
        }
        else
        {
            returned = this.splineVector[i].evaluate( z - x[i - 1] );
        }
        if( Double.isNaN( returned ) || returned < 0 )
        {
            returned = 0;
        }
        else if( Double.isInfinite( returned ) || returned > 1 )
        {
            returned = 1;
        }
        return returned;
    }

    /**
     * Returns the index of <SPAN CLASS="MATH"><I>P</I></SPAN>, the {@link Polynomial} instance used to evaluate <SPAN
     * CLASS="MATH"><I>x</I></SPAN>, in an <TT>ArrayList</TT> table instance returned by
     * <TT>getSplinePolynomials()</TT>. This index <SPAN CLASS="MATH"><I>k</I></SPAN> gives also the interval in table
     * <SPAN CLASS="textbf">X</SPAN> which contains the value <SPAN CLASS="MATH"><I>x</I></SPAN> (i.e. such that <SPAN
     * CLASS="MATH"><I>x</I><SUB>k</SUB> &lt; <I>x</I>&nbsp;&lt;=&nbsp;<I>x</I><SUB>k+1</SUB></SPAN>).
     *
     * @return Index of the polynomial check with x in the Polynomial list returned by methodgetSplinePolynomials
     *
     */
    public int getFitPolynomialIndex( double x )
    {
        // Algorithme de recherche binaire legerement modifie
        int j = this.x.length - 1;
        if( x > this.x[j] )
        {
            return j + 1;
        }
        int tmp = 0;
        int i = 0;
        while( ( i + 1 ) != j )
        {
            if( x > this.x[tmp] )
            {
                i = tmp;
                tmp = i + ( ( j - i ) / 2 );
            }
            else
            {
                j = tmp;
                tmp = i + ( ( j - i ) / 2 );
            }
            if( j == 0 ) // le point est < a x_0, on sort
            {
                i--;
            }
        }
        return i + 1;
    }

    private void resolve()
    {
        /*
         taken from D.S.G Pollock's paper, "Smoothing with Cubic Splines",
         Queen Mary, University of London (1993)
         http://www.qmw.ac.uk/~ugte133/PAPERS/SPLINES.PDF
         */
        int n = this.x.length;
        double[] h = new double[n];
        double[] r = new double[n];
        double[] u = new double[n];
        double[] v = new double[n];
        double[] w = new double[n];
        double[] q = new double[n + 1];
        double[] sigma = new double[this.weight.length];

        for( int i = 0; i < sigma.length; i++ )
        {
            if( this.weight[i] <= 0 )
            {
                sigma[i] = 1.0e100; // arbitrary large number to avoid 1/0
            }
            else
            {
                sigma[i] = 1 / Math.sqrt( this.weight[i] );
            }
        }

        double mu;
        n = this.x.length - 1;
        if( this.rho <= 0 )
        {
            mu = 1.0e100; // arbitrary large number to avoid 1/0
        }
        else
        {
            mu = ( 2 * ( 1 - this.rho ) ) / ( 3 * this.rho );
        }

        h[0] = this.x[1] - this.x[0];
        r[0] = 3 / h[0];
        for( int i = 1; i < n; i++ )
        {
            h[i] = this.x[i + 1] - this.x[i];
            r[i] = 3 / h[i];
            q[i] = ( 3 * ( this.y[i + 1] - y[i] ) / h[i] ) - ( 3 * ( this.y[i] - this.y[i - 1] ) / h[i - 1] );
        }

        for( int i = 1; i < n; i++ )
        {
            u[i] = ( r[i - 1] * r[i - 1] * sigma[i - 1] ) + ( ( r[i - 1] + r[i] ) * ( r[i - 1] + r[i] ) * sigma[i] )
                   + ( r[i] * r[i] * sigma[i + 1] );
            u[i] = mu * u[i] + 2 * ( this.x[i + 1] - this.x[i - 1] );
            v[i] = ( -( r[i - 1] + r[i] ) * r[i] * sigma[i] ) - ( r[i] * ( r[i] + r[i + 1] ) * sigma[i + 1] );
            v[i] = ( mu * v[i] ) + h[i];
            w[i] = mu * r[i] * r[i + 1] * sigma[i + 1];
        }
        q = Quincunx( u, v, w, q );

        // extrapolation a gauche
        double[] params = new double[4];
        params[0] = this.y[0] - ( mu * r[0] * q[1] * sigma[0] );
        double dd1 = this.y[1] - ( mu * ( ( -r[0] - r[1] ) * q[1] + r[1] * q[2] ) * sigma[1] );
        params[1] = ( dd1 - params[0] ) / h[0] - ( q[1] * h[0] / 3 );
        this.splineVector[0] = new SmoothingPolynomial( params );

        // premier polynome
        params[0] = this.y[0] - ( mu * r[0] * q[1] * sigma[0] );
        double dd2 = this.y[1] - ( mu * ( ( -r[0] - r[1] ) * q[1] + r[1] * q[2] ) * sigma[1] );
        params[3] = q[1] / ( 3 * h[0] );
        params[2] = 0;
        params[1] = ( ( dd2 - params[0] ) / h[0] ) - ( q[1] * h[0] / 3 );
        this.splineVector[1] = new SmoothingPolynomial( params );

        // les polynomes suivants
        int j;
        for( j = 1; j < n; j++ )
        {
            params[3] = ( q[j + 1] - q[j] ) / ( 3 * h[j] );
            params[2] = q[j];
            params[1] = ( ( q[j] + q[j - 1] ) * h[j - 1] ) + this.splineVector[j].getCoefficient( 1 );
            params[0] = ( r[j - 1] * q[j - 1] ) + ( ( -r[j - 1] - r[j] ) * q[j] ) + ( r[j] * q[j + 1] );
            params[0] = this.y[j] - ( mu * params[0] * sigma[j] );
            this.splineVector[j + 1] = new SmoothingPolynomial( params );
        }

        // extrapolation a droite
        j = n;
        params[3] = 0;
        params[2] = 0;
        params[1] = this.splineVector[j].derivative( this.x[n] - this.x[n - 1] );
        params[0] = this.splineVector[j].evaluate( this.x[n] - this.x[n - 1] );
        this.splineVector[n + 1] = new SmoothingPolynomial( params );
    }

    private static double[] Quincunx( double[] u, double[] v, double[] w, double[] q )
    {
        u[0] = 0;
        v[1] /= u[1];
        w[1] /= u[1];
        int j;
        for( j = 2; j < ( u.length - 1 ); j++ )
        {
            u[j] = u[j] - ( u[j - 2] * w[j - 2] * w[j - 2] ) - ( u[j - 1] * v[j - 1] * v[j - 1] );
            v[j] = ( v[j] - ( u[j - 1] * v[j - 1] * w[j - 1] ) ) / u[j];
            w[j] /= u[j];
        }

        // forward substitution
        q[1] -= v[0] * q[0];
        for( j = 2; j < ( u.length - 1 ); j++ )
        {
            q[j] = q[j] - ( v[j - 1] * q[j - 1] ) - ( w[j - 2] * q[j - 2] );
        }
        for( j = 1; j < ( u.length - 1 ); j++ )
        {
            q[j] /= u[j];
        }

        // backward substitution
        q[u.length - 1] = 0;
        for( j = ( u.length - 3 ); j > 0; j-- )
        {
            q[j] = q[j] - ( v[j] * q[j + 1] ) - ( w[j] * q[j + 2] );
        }

        return q;
    }
}
