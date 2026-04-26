/*******************************************************************************************
 *   AUTHOR: Dr Michael Thomas Flanagan
 *   DATE:   22 April 2004
 *   UPDATE: 21 November 2006, 31 December 2006, 14 April 2007, 19 October 2007, 16-29 March 2008,
 *           3 July 2008, 19 September 2008, 28 September 2008, 13 and 18 October 2009
 *
 *   DOCUMENTATION:
 *   See Michael Thomas Flanagan's Java library on-line web page:
 *   http://www.ee.ucl.ac.uk/~mflanaga/java/PsRandom.html
 *   http://www.ee.ucl.ac.uk/~mflanaga/java/
 *
 * Copyright (c) 2004 - 2009  Michael Thomas Flanagan
 *
 * PERMISSION TO COPY:
 *
 * Permission to use, copy and modify this software and its documentation for NON-COMMERCIAL purposes is granted, without fee,
 * provided that an acknowledgement to the author, Dr Michael Thomas Flanagan at www.ee.ucl.ac.uk/~mflanaga, appears in all copies
 * and associated documentation or publications.
 *
 * Redistributions of the source code of this source code, or parts of the source codes, must retain the above copyright notice, this list of conditions
 * and the following disclaimer and requires written permission from the Michael Thomas Flanagan:
 *
 * Redistribution in binary form of all or parts of this class must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided with the distribution and requires written permission from the Michael Thomas Flanagan:
 *
 * Dr Michael Thomas Flanagan makes no representations about the suitability or fitness of the software for any or for a particular purpose.
 * Dr Michael Thomas Flanagan shall not be liable for any damages suffered as a result of using, modifying or distributing this software
 * or its derivatives.
 *
 ***************************************************************************************/
package agents.anac.y2015.agentBuyogV2.flanagan.math;

public class LogNormalThreeParFunctData {
  public double cfd;
  public double alpha;
  public double beta;
  public double gamma;

  public LogNormalThreeParFunctData(double cfd, double alpha, double beta, double gamma) {
    this.cfd = cfd;
    this.alpha = alpha;
    this.beta = beta;
    this.gamma = gamma;
  }
}