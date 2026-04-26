package agents.anac.y2019.harddealer;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import agents.anac.y2019.harddealer.math3.optimization.GoalType;
import agents.anac.y2019.harddealer.math3.optimization.PointValuePair;
import agents.anac.y2019.harddealer.math3.optimization.linear.LinearConstraint;
import agents.anac.y2019.harddealer.math3.optimization.linear.LinearObjectiveFunction;
import agents.anac.y2019.harddealer.math3.optimization.linear.Relationship;
import agents.anac.y2019.harddealer.math3.optimization.linear.SimplexSolver;
import genius.core.Bid;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.BoaParty;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.NegotiationInfo;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.utility.AbstractUtilitySpace;


@SuppressWarnings({"serial", "deprecation"})
public class HardDealer extends BoaParty
{
	@Override
	public void init(NegotiationInfo info)
	{
		// The choice for each component is made here
		AcceptanceStrategy 	ac  = new HardDealer_AS();
		OfferingStrategy 	os  = new HardDealer_BS();
		OpponentModel 		om  = new HardDealer_OM();
		OMStrategy			oms = new HardDealer_OMS();

		// All component parameters can be set below.
		Map<String, Double> noparams = Collections.emptyMap();
		Map<String, Double> osParams = new HashMap<String, Double>();
		// Set the concession parameter "e" for the offering strategy to yield Boulware-like behavior
		osParams.put("e", 1.8 / info.getDeadline().getTimeOrDefaultTimeout());

		// Initialize all the components of this party to the choices defined above
		configure(ac, noparams,
				os,	osParams,
				om, noparams,
				oms, noparams);
		super.init(info);
	}

    // total number of values
    private int nValues;
    // occurrences of the values in bids
    private Integer[] occurenceCheck;
    
	@Override
    public AbstractUtilitySpace estimateUtilitySpace() {		
		AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(getDomain());
        List<IssueDiscrete> issues = additiveUtilitySpaceFactory.getIssues();

        List<Bid> ranking = userModel.getBidRanking().getBidOrder();
        double maxUtil = userModel.getBidRanking().getHighUtility();
        double minUtil = userModel.getBidRanking().getLowUtility();
        
     // The linear method estimates the utilities but does nothing with the issue weights.
        // We use a variance and spread based method to estimate these issue weights. 
      		List<Bid> bidOrder = userModel.getBidRanking().getBidOrder();

    		/**
    		 * Create a custom issue weight
    		 * Iterate over all values in all bids to build up the issue weights.
    		 */
    		
    		int bidOrderSize = bidOrder.size();
    		int numberOfIssues = 0;
    		// Initialize dictionaries in which we will store information we will extract from the bidranking.
    		// Information for the Values
    		// v1. Store at which index each value occurs in the bidranking
    		// v2. Store the mean index in the ranking of each value
    		// v3. Store the variance for each value
    		Map<String, ArrayList<Integer>> valueIndicesListDict = new HashMap<String, ArrayList<Integer>>();
    		Map<String, Double> valueMeanIndexDict = new HashMap<String, Double>();
    		Map<String, Double> valueVarianceDict = new HashMap<String, Double>();	
    		
    		// Information for the Issues
    		// i1. Store the variance for each issue (combined value variance)
    		// i2. Store the spread of each issue (how far apart are the mean indices of its values)
    		// i3. Store the Weight of the Issues
    		Map<Issue, Double> issueVarianceDict = new HashMap<Issue, Double>();
    		Map<Issue, Double> issueSpreadDict = new HashMap<Issue, Double>();
    		Map<Issue, Double> issueWeightDict = new HashMap<Issue, Double>();

    		// Variable that can be tweaked:
    		// Tweaks how much the weights are shifted when converting from variance to partial issue weight.
    		double inverseShiftVariable = 2;
    		
    		// Tweaks the ratio between the usage of the variance and the spread of the means within an issue to build the issue weight.
    		// This variable is scaled automatically dependinng on the size of the bidordering.
    		// The transition value of 40 bids in the bidranking has been chosen trough extensive testing.
    		double blendVariable = 1-(Math.min(bidOrderSize/40,1));

    		// Create a list of all issuevalue strings. This is used for looping over all values in the dictionary.
    		// Initialize the valueIndicesListDict with all the values and initialize empty ArrayLists.
    		ArrayList<String> listOfAllIssueValues = new ArrayList<String>();
    		for (Issue i : issues)
    		{
    			List<ValueDiscrete> values = ((IssueDiscrete) i).getValues();
    			for (ValueDiscrete v : values)
    			{
    				ArrayList<Integer> indicesList = new ArrayList<Integer>();
    				String IssueValueString =  i.getName() +v.toString();

    				valueIndicesListDict.put(IssueValueString , indicesList);
    				listOfAllIssueValues.add(IssueValueString);
    			}
    			numberOfIssues += 1;
    		}

    // 1. Build up the double representation of the issue weights by using the variance and the spread.
    		// Build the valueIndicesListDict by iterating trough all bids and storing the location of the values.
    		int bidIndex = 0;
    		for (Bid bid : bidOrder)
    		{
    			for (Issue i : issues)
    			{
    				// Fill the valueIndicesListDict with the indices at which these values occur.
    				int no = i.getNumber();
    				ValueDiscrete value = (ValueDiscrete) bid.getValue(no);
    				String IssueValueString =  i.getName() +value.toString();
    				ArrayList<Integer> indicesList = valueIndicesListDict.get(IssueValueString);
    				indicesList.add(bidIndex);
    				valueIndicesListDict.replace(IssueValueString,indicesList);
    			}
    			// Setup for the next bid in the loop
    			bidIndex += 1;
    		}

    		// Calculate variance for each value.
    		for (String key : listOfAllIssueValues)
    		{
    			ArrayList<Integer> indicesList = valueIndicesListDict.get(key);
    			int sumOfIndices = indicesList.stream().mapToInt(Integer::intValue).sum();
    			double mu;
    			if (sumOfIndices == 0)
    			{
    				mu = bidOrderSize/2;
    			} 
    			else 
    			{
    				mu = sumOfIndices / indicesList.size();
    			}
    			valueMeanIndexDict.put(key,mu);
    			double P = ((double) 1) / indicesList.size();

    			ArrayList<Double> listOfPartialVariance = new ArrayList<Double>();
    			for (int index : indicesList)
    			{
    				double partialVariance = Math.pow((index - mu),2.0) * P;
    				listOfPartialVariance.add(partialVariance);
    			}

    			double variance = listOfPartialVariance.stream().reduce(0.0, Double::sum);
    			valueVarianceDict.put(key,variance);
    		}

    		// Add the variances of the values together to get issue variances.
    		// Determine the spread of the mean values within the issues.
    		for (Issue i : issues)
    		{
    			double issueVariance = 0;
    			double issueSpread = 0;
    			List<ValueDiscrete> values = ((IssueDiscrete) i).getValues();
    			for (ValueDiscrete v : values)
    			{
    				String IssueValueString = i.getName()+v.toString();
    				double middleIndex = bidOrderSize/2;
    				double meanIndex = valueMeanIndexDict.get(IssueValueString);
    				double distanceFromMiddleIndex = Math.abs(meanIndex - middleIndex);
    				issueSpread += distanceFromMiddleIndex;
    				issueVariance += valueVarianceDict.get(IssueValueString);
    			}
    			issueSpreadDict.put(i, issueSpread);
    			issueVarianceDict.put(i,issueVariance);
    		}

    		// Normalize the spread
    		double totalSpread = 0;
    		for (Issue i : issues)
    		{
    			totalSpread += issueSpreadDict.get(i);
    		}
    		for (Issue i : issues)
    		{
    			double spread = issueSpreadDict.get(i);
    			double normalizedSpread = spread/totalSpread;
    			issueSpreadDict.replace(i,normalizedSpread);
    		}

    		// Calculate the max variance
    		double oldValue = 0;
    		double maxVariance = 0;
    		for (Issue i : issues)
    		{
    			double newValue = issueVarianceDict.get(i);
    			maxVariance = Math.max(newValue,oldValue);
    			oldValue = newValue;
    		}

    		// Normalize the variance and translate it to a partial issueWeight
    		double totalIssueWeight = 0;
    		for (Issue i : issues)
    		{
    			double oldVariance = issueVarianceDict.get(i);
    			double partialIssueWeight = Math.abs(inverseShiftVariable - oldVariance/maxVariance);
    			totalIssueWeight += partialIssueWeight;
    			issueWeightDict.put(i,partialIssueWeight);
    		}

    		// Normalize the partial Issue weights
    		for (Issue i: issues)
    		{
    			double oldWeight = issueWeightDict.get(i);
    			double newWeight = oldWeight/totalIssueWeight;
    			issueWeightDict.replace(i,newWeight);
    		}
    		// Combine spread and variance into one issue weight and load into Genius. Use a blendVariable to adjust their ratio
    		for (Issue i : issues)
    		{
    			double finalIssueWeight = (issueWeightDict.get(i) * blendVariable) + (issueSpreadDict.get(i) * (1 - blendVariable));
    	    	issueWeightDict.replace(i, finalIssueWeight);
    			additiveUtilitySpaceFactory.setWeight(i, finalIssueWeight);
    		}

// The issue weights are now constructed, now we are going to generate the weights of the values trough linear programming.
        // All values are collected in a LinkedHashMap
        LinkedHashMap<IssueDiscrete, List<ValueDiscrete>> valuesMap = new LinkedHashMap<>();

        nValues = 0;
        for (IssueDiscrete issue : issues) {
            valuesMap.put(issue, issue.getValues());
            nValues += issue.getValues().size();
        }

        occurenceCheck = new Integer[nValues];
        Arrays.fill(occurenceCheck,0);

        int nSlackVariables = ranking.size()-1;
        // The number of variables needed for the linear optimization
        int nVariables = nSlackVariables + nValues;

        // The objective function is to minimize all slack variables
        // Therefore, all the slack variables should have a coefficient of 1, and all the other should have a coefficient of 0
        double[] functionList = new double[nVariables];
        Arrays.fill(functionList, 0);

        for (int i = 0; i < nSlackVariables; i++) {
            functionList[i] = 1;
        }

        LinearObjectiveFunction f = new LinearObjectiveFunction(functionList, 0);

        // A collection with all constraints
        Collection<LinearConstraint> constraints = new ArrayList<>();
        createVarGEQ0Constraints(constraints, functionList, nSlackVariables);
        createSlackComparisonGEQ0Constraints(constraints, functionList, nSlackVariables, valuesMap, ranking);
        createMaxMinConstraints(constraints, valuesMap, ranking, nSlackVariables, maxUtil, minUtil);

        
        // Use a Simplex solver to solve f
        SimplexSolver solver = new SimplexSolver();
        solver.setMaxIterations(Integer.MAX_VALUE);
        PointValuePair solution = solver.optimize(f, constraints, GoalType.MINIMIZE, true);

        // The average utility of a value is estimated based on the max utility, min utility and the variance within the bid ranking.
		double functionIndex = totalIssueWeight/numberOfIssues;
		double exponent = 5 - (4/(1+Math.exp(-50*(functionIndex-1))));
		double yScale = maxUtil - minUtil;
        
        double average = minUtil + yScale*Math.pow(0.5, exponent);
        // Initialization of new utilities
        // The utilities of the actual values are stored at indices after the slack variables,
        // so start iterating after nSlackVariables
        int iterator = nSlackVariables;
        for (IssueDiscrete issue : issues) {
            // In case there is not much information about the value (0, 1 or 2 occurrences), it also takes the average into consideration
            for (int v = 0; v < issue.getNumberOfValues(); v++) {
                if(occurenceCheck[iterator - nSlackVariables] < 3) {
                    int occ = occurenceCheck[iterator - nSlackVariables];
                    double util = ((occ * solution.getPoint()[iterator]) + average) / (occ + 1);                
                    additiveUtilitySpaceFactory.setUtility(issue, issue.getValue(v), util);
                }
                else {
                    additiveUtilitySpaceFactory.setUtility(issue, issue.getValue(v), solution.getPoint()[iterator]);
                }

                iterator++;
            }
        }
        // Normalize the weights
        additiveUtilitySpaceFactory.normalizeWeights();
    
        // The factory is done with setting all parameters, now return the estimated utility space
        return additiveUtilitySpaceFactory.getUtilitySpace();
    }

	
    // Transforms a bid into a double list of 0's and 1's representing it values
    private double[] valuesFunctionList(Bid bid, LinkedHashMap<IssueDiscrete, List<ValueDiscrete>> values, int nSlackValues) {
        double[] linear = new double[nSlackValues + nValues];
        Arrays.fill(linear, 0);

        int count = 0;

        // A 1 is placed at the corresponding value position
        for (int i = 0; i < bid.getIssues().size(); i++) {
            ValueDiscrete v = (ValueDiscrete) bid.getValue(i+1);

            for (int val = 0; val < values.get(bid.getIssues().get(i)).size(); val++) {
                ValueDiscrete v2 = values.get(bid.getIssues().get(i)).get(val);
                if (v.equals(v2)) {
                    linear[count + nSlackValues] = 1;
                    occurenceCheck[count] += 1;
                }
                count++;
            }
        }
        return linear;
    }

    // Creates a 'variable is greater or equal than 0' constraint for all slack and value variables
    private void createVarGEQ0Constraints(Collection<LinearConstraint> constraints, double[] functionList, int nSlackVariables) {
        for (int i = 0; i < functionList.length; i++) {
        	// reset all coefficients to 0
            Arrays.fill(functionList, 0);
            // set the coefficient for variable i to 1
            functionList[i] = 1;
            // create the greater or equal constraint
            constraints.add(new LinearConstraint(functionList, Relationship.GEQ, 0));
        }
    }

    // Creates the 'slack + comparison variable is greater or equal than 0' constraints for every slack and corresponding comparison variable
    private void createSlackComparisonGEQ0Constraints(Collection<LinearConstraint> constraints, double[] functionList, int nSlackVariables, LinkedHashMap<IssueDiscrete, List<ValueDiscrete>> values, List<Bid> ranking) {
        for (int i = 0; i < nSlackVariables; i++) {
        	// reset all coefficients to 0
            Arrays.fill(functionList, 0);
            // get bid o
            Bid o = ranking.get(i);
            // get bid o'
            Bid oPrime = ranking.get(i+1);
            // create a function list for o
            double[] oList = valuesFunctionList(o, values, nSlackVariables);
            // create a function list for o'
            double[] oPrimeList = valuesFunctionList(oPrime, values, nSlackVariables);
            // create the delta u as values of o - values of o'
            for(int j = 0; j < functionList.length; j++) {
            	functionList[j] = oPrimeList[j] - oList[j];
            }
            // set the coefficient of slack variable i to 1
            functionList[i] = 1;
            constraints.add(new LinearConstraint(functionList, Relationship.GEQ, 0));
        }
    }

    // Creates the constraints which represent the max and min utility
    private void createMaxMinConstraints(Collection<LinearConstraint> constraints,
                                                  LinkedHashMap<IssueDiscrete, List<ValueDiscrete>> values, List<Bid> ranking, int nSlackVariables, double maxU, double minU) {
        
    	Bid strongestBid = ranking.get(ranking.size()-1);
        double[] functionList = valuesFunctionList(strongestBid, values, nSlackVariables);
        // The comparison strongest bid == maxU is added
        constraints.add(new LinearConstraint(functionList, Relationship.EQ, maxU));
        
    	Bid weakestBid = ranking.get(0);
        functionList = valuesFunctionList(weakestBid, values, nSlackVariables);
        // The comparison strongest bid == maxU is added
        constraints.add(new LinearConstraint(functionList, Relationship.EQ, minU));

    }

    
	@Override
	public String getDescription()
	{
		return "Hardheaded but concedes at the very end to make a deal.";
	}
	// All the rest of the agent functionality is defined by the components selected above, using the BOA framework
}
