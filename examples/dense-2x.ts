import _ from 'lodash';
import { Model, Dense, activation } from '../src';

const model = new Model({ seed: 'testing-1234' });

model.add(new Dense({ units: 4, activation: activation.ReLU }));
model.add(new Dense({ units: 4, activation: 'relu' }));
model.add(new Dense({ units: 1, activation: 'sigmoid' }));


const samples = _.times(20, n => ({ x: n, y: n * 2 }));


model.input(samples);


model.compile();

model.fit(samples);
