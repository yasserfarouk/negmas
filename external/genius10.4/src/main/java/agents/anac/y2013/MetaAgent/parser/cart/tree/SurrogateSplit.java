package agents.anac.y2013.MetaAgent.parser.cart.tree;

public class SurrogateSplit extends Split {
	String _name;
	double _value;
	PrimarySplit.Direction _direction;
	double _agree;
	double _adj;
	int _splitsCount;

private SurrogateSplit(String name, double value, PrimarySplit.Direction direction,
			double agree, double adj, int splitsCount) {
		super(name,value,direction);
		this._agree = agree;
		this._adj = adj;
		this._splitsCount = splitsCount;
	}


public static SurrogateSplit factory(String text){
	text=text.trim();
	String[]wordstext=text.split(" +");
	String name=wordstext[0];
	double value=Double.parseDouble(wordstext[2]);
	PrimarySplit.Direction dir;
	if(wordstext[5].contains("left"))
		dir=PrimarySplit.Direction.LEFT;
	else
		dir=PrimarySplit.Direction.RIGHT;

	
	double agree=Double.parseDouble(Node.substring(text,"agree=",","));
	double adj=Double.parseDouble(Node.substring(text,"adj=",","));
	int count=Integer.parseInt(Node.substring(text, "(",text.indexOf("split")).trim());
	SurrogateSplit s=new SurrogateSplit(name, value, dir, agree, adj, count);
	return s;
	
}

@Override
public String toString() {
	return "SurrogateSplit [_name=" + _name + ", _value=" + _value
			+ ", _direction=" + _direction + ", _agree=" + _agree + ", _adj="
			+ _adj + ", _splitsCount=" + _splitsCount + "]";
}
}