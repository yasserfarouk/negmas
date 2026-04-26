package agents.anac.y2013.MetaAgent.agentsData.agents;

import agents.anac.y2013.MetaAgent.agentsData.AgentData;


public class DataCUHKAgent extends AgentData {
	String data="Node number 1: 10078 observations,    complexity param=0.03558322\n  mean=0.07454525, MSE=0.02117389 \n  left son=2 (3358 obs) right son=3 (6720 obs)\n  Primary splits:\n      ReservationValue          < 0.375     to the right, improve=0.035583220, (0 missing)\n      RelevantStdevU            < 0.1257632 to the right, improve=0.005655008, (0 missing)\n      AvgUtilStdev              < 0.3419255 to the right, improve=0.004750698, (0 missing)\n      DomainSize                < 7         to the left,  improve=0.004750698, (0 missing)\n      UtilityOfFirstOpponentBid < 0.3308333 to the left,  improve=0.004724202, (0 missing)\n\nNode number 2: 3358 observations\n  mean=0.03571526, MSE=0.01492362 \n\nNode number 3: 6720 observations\n  mean=0.09394869, MSE=0.02316724 \n\n";
	public String getText() {
		return data;
	}
}
