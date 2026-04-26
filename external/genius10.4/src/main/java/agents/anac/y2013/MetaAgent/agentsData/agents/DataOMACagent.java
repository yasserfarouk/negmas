package agents.anac.y2013.MetaAgent.agentsData.agents;

import agents.anac.y2013.MetaAgent.agentsData.AgentData;


public class DataOMACagent extends AgentData {
	String data="Node number 1: 10071 observations,    complexity param=0.01462601\n  mean=0.06507811, MSE=0.02149636 \n  left son=2 (3287 obs) right son=3 (6784 obs)\n  Primary splits:\n      DiscountFactor   < 0.625     to the left,  improve=0.014626010, (0 missing)\n      ReservationValue < 0.375     to the right, improve=0.012594450, (0 missing)\n      RelevantEU       < 0.6151587 to the right, improve=0.011290560, (0 missing)\n      DomainSize       < 64260.5   to the left,  improve=0.011253840, (0 missing)\n      AvgUtil          < 0.6824029 to the right, improve=0.008979756, (0 missing)\n\nNode number 2: 3287 observations\n  mean=0.03960462, MSE=0.02153621 \n\nNode number 3: 6784 observations,    complexity param=0.01286791\n  mean=0.07742059, MSE=0.02101031 \n  left son=6 (4128 obs) right son=7 (2656 obs)\n  Primary splits:\n      RelevantEU                < 0.6151587 to the right, improve=0.019544620, (0 missing)\n      ReservationValue          < 0.375     to the right, improve=0.015447620, (0 missing)\n      UtilityOfFirstOpponentBid < 0.2920833 to the right, improve=0.014484860, (0 missing)\n      DomainSize                < 64260.5   to the left,  improve=0.008853446, (0 missing)\n      RelevantStdevU            < 0.1075864 to the left,  improve=0.008242062, (0 missing)\n  Surrogate splits:\n      UtilityOfFirstOpponentBid < 0.3408333 to the right, agree=0.845, adj=0.605, (0 split)\n      AvgUtil                   < 0.5191162 to the right, agree=0.835, adj=0.578, (0 split)\n      RelevantStdevU            < 0.1408769 to the left,  agree=0.742, adj=0.342, (0 split)\n      AvgUtilStdev              < 0.2071902 to the left,  agree=0.712, adj=0.263, (0 split)\n      numOfIssues               < 2         to the right, agree=0.691, adj=0.210, (0 split)\n\nNode number 6: 4128 observations\n  mean=0.06116608, MSE=0.01122019 \n\nNode number 7: 2656 observations\n  mean=0.1026836, MSE=0.03517743 \n\n";
	public String getText() {
		return data;
	}
}
