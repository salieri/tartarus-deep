import _ from 'lodash';

import {
  DeferredInputFeed,
  EndOfStreamException,
  InputFeedRecord,
  Sample,
  Label,
} from './feed';

import { NDArray, NDArrayCollection } from '../math';
import { DeferredInputCollection, DeferredCollection } from '../nn';


export type ProtoSampleData = NDArray|NDArrayCollection|number[];

export interface ProtoSample {
  x: ProtoSampleData;
  y?: ProtoSampleData;
}


export class MemoryInputFeed extends DeferredInputFeed {
  protected offsetPtr: number = 0;

  protected readonly samples: Sample[];

  protected labels?: Label[];

  public constructor(samples: Sample[]|ProtoSample[]|ProtoSampleData[] = [], labels?: Label[]|ProtoSampleData[]) {
    super();

    if ((labels) && (samples.length !== labels.length)) {
      throw new Error('Sample and label lengths must match');
    }

    this.samples = this.prepareSamples(samples);

    if (labels) {
      this.labels = this.prepareLabels(labels);
    } else if (this.hasLabelData(samples)) {
      this.labels = this.prepareLabels(samples as ProtoSample[]);
    }

    // this is a different test than the one above
    if ((this.labels) && (this.samples.length !== this.labels.length)) {
      throw new Error('Sample and label lengths must match');
    }
  }


  protected hasLabelData(samples: Sample[]|ProtoSample[]|ProtoSampleData[]): boolean {
    return _.every(
      samples, (s: any) => ('y' in s),
    );
  }


  protected prepareSamples(samples: Sample[]|ProtoSample[]|ProtoSampleData[]): Sample[] {
    return _.map(
      samples as any,
      (sample: Sample|ProtoSample|ProtoSampleData) => {
        if ('x' in sample) {
          return { raw: MemoryInputFeed.prepareCollection(sample.x) };
        }

        if ('raw' in sample) {
          return sample as Sample;
        }

        return { raw: MemoryInputFeed.prepareCollection(sample) };
      },
    );
  }


  protected prepareLabels(labels?: Label[]|ProtoSample[]|ProtoSampleData[]): Label[] {
    if (!labels) {
      return [];
    }

    return _.map(
      labels as any,
      (label: Label|ProtoSampleData) => {
        if ('raw' in label) {
          return label as Label;
        }

        if ('y' in label) {
          return { raw: MemoryInputFeed.prepareCollection(label.y) };
        }

        return { raw: MemoryInputFeed.prepareCollection(label) };
      },
    );
  }


  public add(sample: ProtoSampleData, label?: ProtoSampleData): MemoryInputFeed {
    this.samples.push({ raw: MemoryInputFeed.prepareCollection(sample) });

    if (!this.labels) {
      this.labels = [];
    }

    if (label) {
      this.labels.push({ raw: MemoryInputFeed.prepareCollection(label) });
    }

    return this;
  }


  public count(): number {
    return this.samples.length;
  }


  public offset(): number {
    return this.offsetPtr;
  }


  public async next(): Promise<InputFeedRecord> {
    const offsetPtr = this.offsetPtr;

    if (offsetPtr >= this.samples.length) {
      throw new EndOfStreamException();
    }

    const data: InputFeedRecord = {
      name: this.samples[offsetPtr].name || `sample-${offsetPtr}`,
      sample: this.samples[offsetPtr],
      label: this.labels ? this.labels[offsetPtr] : undefined,
      offset: offsetPtr,
      count: this.count(),
    };

    this.offsetPtr += 1;

    return data;
  }


  public async seek(offset: number): Promise<void> {
    if ((offset < 0) || (offset >= this.count())) {
      throw new Error('Offset out of range');
    }

    if (!(Number.isInteger(offset))) {
      throw new Error('Invalid offset');
    }

    this.offsetPtr = offset;
  }


  public hasLabels(): boolean {
    return !!this.labels;
  }


  protected static prepareCollection(sampleData: ProtoSampleData): DeferredInputCollection {
    const sampleCollection = new DeferredCollection();
    const finalSampleData = _.isArray(sampleData) ? new NDArray(sampleData as number[]) : sampleData;

    if (finalSampleData instanceof NDArray) {
      sampleCollection.declareDefault(finalSampleData.getDims());
      sampleCollection.setDefaultValue(finalSampleData);
    } else {
      _.each(
        finalSampleData,
        (val: NDArray, key: string) => {
          sampleCollection.declare(key, val.getDims());
          sampleCollection.setValue(key, val);
        },
      );
    }

    return new DeferredInputCollection(sampleCollection);
  }


  public static factory(protoSamples: ProtoSample[]): MemoryInputFeed {
    const samples: Sample[] = [];
    const labels: Label[] = [];

    _.each(
      protoSamples,
      (proto: ProtoSample) => {
        samples.push({ raw: MemoryInputFeed.prepareCollection(proto.x) });

        if (proto.y) {
          labels.push({ raw: MemoryInputFeed.prepareCollection(proto.y) });
        }
      },
    );

    return new MemoryInputFeed(samples, (labels.length > 0) ? labels : undefined);
  }
}
