package genius.core.qualitymeasures;

import java.util.ArrayList;

/**
 * Simple class to hold the results of the opponent model measures.
 * 
 * @author Mark Hendrikx
 */
public class OpponentModelMeasuresResults {
	
	private ArrayList<Double> timePointList;
    private ArrayList<Double> pearsonCorrelationCoefficientOfBidsList;
    private ArrayList<Double> rankingDistanceOfBidsList;
    private ArrayList<Double> rankingDistanceOfIssueWeightsList;
    private ArrayList<Double> averageDifferenceBetweenBidsList;
    private ArrayList<Double> averageDifferenceBetweenIssueWeightsList;
    private ArrayList<Double> kalaiDistanceList;
    private ArrayList<Double> nashDistanceList;
    private ArrayList<Double> averageDifferenceOfParetoFrontierList;
    private ArrayList<Double> percentageOfCorrectlyEstimatedParetoBidsList;
    private ArrayList<Double> percentageOfIncorrectlyEstimatedParetoBidsList;
    private ArrayList<Double> paretoFrontierDistanceList;
    private ArrayList<Double> bidIndices;
    private double[][] empty;

    public OpponentModelMeasuresResults() {
    	timePointList = new ArrayList<Double>();
    	pearsonCorrelationCoefficientOfBidsList = new ArrayList<Double>();
    	rankingDistanceOfBidsList = new ArrayList<Double>();
    	rankingDistanceOfIssueWeightsList = new ArrayList<Double>();
    	averageDifferenceBetweenBidsList = new ArrayList<Double>();
    	averageDifferenceBetweenIssueWeightsList = new ArrayList<Double>();
    	kalaiDistanceList = new ArrayList<Double>();
    	nashDistanceList = new ArrayList<Double>();
    	averageDifferenceOfParetoFrontierList = new ArrayList<Double>();
    	percentageOfCorrectlyEstimatedParetoBidsList = new ArrayList<Double>();
    	percentageOfIncorrectlyEstimatedParetoBidsList = new ArrayList<Double>();
    	paretoFrontierDistanceList = new ArrayList<Double>();
    	bidIndices = new ArrayList<Double>();
		empty = new double[2][1];
		empty [0][0] = 0;
		empty [1][0] = 0;
    }

    public void addTimePoint(double timePoint) {
    	timePointList.add(timePoint);
    }
	
    public void addPearsonCorrelationCoefficientOfBids(double pearsonCorrelationCoefficientOfBids) {
    	pearsonCorrelationCoefficientOfBidsList.add(pearsonCorrelationCoefficientOfBids);
    }
    
    public void addRankingDistanceOfBids(double rankingDistanceOfBids) {
    	rankingDistanceOfBidsList.add(rankingDistanceOfBids);
    }
    
    public void addRankingDistanceOfIssueWeights(double rankingDistanceOfIssueWeights) {
    	rankingDistanceOfIssueWeightsList.add(rankingDistanceOfIssueWeights);
    }
    
    public void addAverageDifferenceBetweenBids(double averageDifferenceBetweenBids) {
    	averageDifferenceBetweenBidsList.add(averageDifferenceBetweenBids);
    }
    
    public void addAverageDifferenceBetweenIssueWeights(double averageDifferenceBetweenIssueWeights) {
    	averageDifferenceBetweenIssueWeightsList.add(averageDifferenceBetweenIssueWeights);
    }
    
    public void addKalaiDistance(double kalaiDistance) {
    	kalaiDistanceList.add(kalaiDistance);
    }
    
    public void addNashDistance(double nashDistance) {
    	nashDistanceList.add(nashDistance);
    }
    
    public void addAverageDifferenceOfParetoFrontier(double averageDifferenceOfParetoFrontier) {
    	averageDifferenceOfParetoFrontierList.add(averageDifferenceOfParetoFrontier);
    }
    
    public void addPercentageOfCorrectlyEstimatedParetoBids(double percentageOfCorrectlyEstimatedParetoBids) {
    	percentageOfCorrectlyEstimatedParetoBidsList.add(percentageOfCorrectlyEstimatedParetoBids);
    }
    
    public void addPercentageOfIncorrectlyEstimatedParetoBids(double percentageOfIncorrectlyEstimatedParetoBids) {
    	percentageOfIncorrectlyEstimatedParetoBidsList.add(percentageOfIncorrectlyEstimatedParetoBids);
    }
    
    public void addBidIndex(int bidIndex) {
    	bidIndices.add((double)bidIndex);
    }

    public void addParetoFrontierDistance(double paretoFrontierDistance) {
    	paretoFrontierDistanceList.add(paretoFrontierDistance);
    }
    
	public ArrayList<Double> getTimePointList() {
		return timePointList;
	}

	public double[][] getPearsonCorrelationCoefficientOfBidsListData() {
		return convertToChartData(timePointList, pearsonCorrelationCoefficientOfBidsList);
	}

	public double[][] getRankingDistanceOfBidsListData() {
		return convertToChartData(timePointList, rankingDistanceOfBidsList);
	}

	public double[][] getRankingDistanceOfIssueWeightsListData() {
		return convertToChartData(timePointList, rankingDistanceOfIssueWeightsList);
	}

	public double[][] getAverageDifferenceBetweenBidsListData() {
		return convertToChartData(timePointList, averageDifferenceBetweenBidsList);
	}

	public double[][] getAverageDifferenceBetweenIssueWeightsListData() {
		return convertToChartData(timePointList, averageDifferenceBetweenIssueWeightsList);
	}

	public double[][] getKalaiDistanceListData() {
		return convertToChartData(timePointList, kalaiDistanceList);
	}

	public double[][] getNashDistanceListData() {
		return convertToChartData(timePointList, nashDistanceList);
	}

	public double[][] getAverageDifferenceOfParetoFrontierListData() {
		return convertToChartData(timePointList, averageDifferenceOfParetoFrontierList);
	}

	public double[][] getPercentageOfCorrectlyEstimatedParetoBidsListData() {
		return convertToChartData(timePointList, percentageOfCorrectlyEstimatedParetoBidsList);
	}

	public double[][] getPercentageOfIncorrectlyEstimatedParetoBidsListData() {
		return convertToChartData(timePointList, percentageOfIncorrectlyEstimatedParetoBidsList);
	}

	public double[][] getParetoFrontierDistanceListData() {
		return convertToChartData(timePointList, paretoFrontierDistanceList);
	}

	private double[][] convertToChartData(ArrayList<Double> timeSnaps, ArrayList<Double> items) {
		if (timeSnaps.size() != items.size()) {
			return empty;
		}

		double[][] data = new double[2][items.size()];
		try
        {
	    	for (int i = 0; i < items.size(); i++) {
	    		data [0][i] = timeSnaps.get(i);
	    		data [1][i] = items.get(i);
	    	}
        } catch (Exception e) {
			e.printStackTrace();
        	return null;
		}
		return data;
	}

	public ArrayList<Double> getPearsonCorrelationCoefficientOfBidsList() {
		return pearsonCorrelationCoefficientOfBidsList;
	}

	public ArrayList<Double> getRankingDistanceOfBidsList() {
		return rankingDistanceOfBidsList;
	}

	public ArrayList<Double> getRankingDistanceOfIssueWeightsList() {
		return rankingDistanceOfIssueWeightsList;
	}

	public ArrayList<Double> getAverageDifferenceBetweenBidsList() {
		return averageDifferenceBetweenBidsList;
	}

	public ArrayList<Double> getAverageDifferenceBetweenIssueWeightsList() {
		return averageDifferenceBetweenIssueWeightsList;
	}

	public ArrayList<Double> getKalaiDistanceList() {
		return kalaiDistanceList;
	}

	public ArrayList<Double> getNashDistanceList() {
		return nashDistanceList;
	}

	public ArrayList<Double> getAverageDifferenceOfParetoFrontierList() {
		return averageDifferenceOfParetoFrontierList;
	}

	public ArrayList<Double> getPercentageOfCorrectlyEstimatedParetoBidsList() {
		return percentageOfCorrectlyEstimatedParetoBidsList;
	}

	public ArrayList<Double> getPercentageOfIncorrectlyEstimatedParetoBidsList() {
		return percentageOfIncorrectlyEstimatedParetoBidsList;
	}

	public ArrayList<Double> getBidIndices() {
		return bidIndices;
	}

	public ArrayList<Double> getParetoFrontierDistanceList() {
		return paretoFrontierDistanceList;
	}
}