/*
*   Diffusion Class
*
*
*   Author:  Dr Michael Thomas Flanagan.
*
*   Created: february 2010
*   Updated:
*
*   DOCUMENTATION:
*   See Michael Thomas Flanagan's Java library on-line web page:
*   http://www.ee.ucl.ac.uk/~mflanaga/java/Diffusion.html
*   http://www.ee.ucl.ac.uk/~mflanaga/java/
*
*   Copyright (c) 2010   Michael Thomas Flanagan
*
*   PERMISSION TO COPY:
*   Permission to use, copy and modify this software and its documentation for
*   NON-COMMERCIAL purposes is granted, without fee, provided that an acknowledgement
*   to the author, Michael Thomas Flanagan at www.ee.ucl.ac.uk/~mflanaga, appears in all copies.
*
*   Dr Michael Thomas Flanagan makes no representations about the suitability
*   or fitness of the software for any or for a particular purpose.
*   Michael Thomas Flanagan shall not be liable for any damages suffered
*   as a result of using, modifying or distributing this software or its derivatives.
*
***************************************************************************************/

package agents.anac.y2015.agentBuyogV2.flanagan.physprop;

import agents.anac.y2015.agentBuyogV2.flanagan.analysis.Stat;
import agents.anac.y2015.agentBuyogV2.flanagan.math.Fmath;

public class Diffusion{


    // METHODS


    // Returns the diffusion coefficient (m^2 s^-1)of a solute assuming a spherical solute
    // molecularWeight  Molecular weight of the solute(Daltons)
    // specificVolume   Specific volume (m^3 kg^-1)
    // viscosity        Solution viscosity (Pa s)
    // concentration	Solute concentration (Molar)
    // temperature      Temperature (degree Celsius)
    public static double diffusionCoefficient(double molecularWeight, double specificVolume, double viscosity, double concentration, double temperature){

	    double tempK = temperature - Fmath.T_ABS;
	    double molecularVolume = molecularWeight*specificVolume/(Fmath.N_AVAGADRO*1000.0);
	    double molecularRadius = Math.pow(3.0*molecularVolume/(4.0*Math.PI),1.0/3.0);
	    double fTerm = 6.0*Math.PI*viscosity*molecularRadius;
	    return  Fmath.K_BOLTZMANN*tempK/fTerm;
    }

    // Returns the number of molecules per square metre in an hexagonally closed-packed monolayer
    // molecularWeight  Molecular weight of the solute(Daltons)
    // specificVolume   Specific volume (m^3 kg^-1)
    public static double planarHexagonalNumberPerSquareMetre(double molecularWeight, double specificVolume){

        double molecularVolume = molecularWeight*specificVolume/(Fmath.N_AVAGADRO*1000.0);
	    double molecularRadius = Math.pow(3.0*molecularVolume/(4.0*Math.PI),1.0/3.0);

	    return  2.0/(3.0*Math.sqrt(3.0)*molecularRadius*molecularRadius);
	}

	// Returns the moles per square metre in an hexagonally closed-packed monolayer
    // molecularWeight  Molecular weight of the solute(Daltons)
    // specificVolume   Specific volume (m^3 kg^-1)
    public static double planarHexagonalMolesPerSquareMetre(double molecularWeight, double specificVolume){

        double molecularVolume = molecularWeight*specificVolume/(Fmath.N_AVAGADRO*1000.0);
	    double molecularRadius = Math.pow(3.0*molecularVolume/(4.0*Math.PI),1.0/3.0);

	    return  2.0/(3.0*Math.sqrt(3.0)*molecularRadius*molecularRadius*Fmath.N_AVAGADRO);
	}

    // Returns the concentration at a distance x from a boundary at time t assuming:
    //  one dimensional difussion
    //  an unchanging concentration at the boundary
    //  zero concentration elsewhere at time zero
    // diffusionCoefficient         Diffusion coefficient (m^2 s^-1)
    // ZeroDistanceConcentration    Concentration at the boundary [x = 0]
    // distance                     Distance, x, from the boundary [x = 0] (m)
    // time                         Time (s)
    public static double oneDimensionalDiffusion(double diffusionCoefficient, double ZeroDistanceConcentration, double distance, double time){

        double arg = distance/(2.0*Math.sqrt(diffusionCoefficient*time));
        return ZeroDistanceConcentration*Stat.erfc(arg);
    }

}
